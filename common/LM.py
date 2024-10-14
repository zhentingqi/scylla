# Licensed under the MIT license.

import torch, math
import torch.nn.functional as F
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from openai import OpenAI
from anthropic import Anthropic
from together import Together
from time import sleep
from random import shuffle


class Txt2TxtGenerator:
    def __init__(
        self,
        model_ckpt: str,
        api: str,
        api_key=None,
        api_key_file="",
        is_chat: bool = True,
        vllm_seed: int = 0,
        vllm_max_model_len=None,
        vllm_use_tqdm=False,
    ):
        assert api in ["hf", "vllm", "openai", "anthropic", "together"]

        self.model_ckpt = model_ckpt
        self.api = api
        self.api_key = api_key
        self.api_key_file = api_key_file
        self.is_chat = is_chat
        self.vllm_seed = vllm_seed
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_use_tqdm = vllm_use_tqdm

        self._load_model()

    def _load_model(self):
        print(f"[LM.py:Txt2TxtGenerator] ==> Loading model {self.model_ckpt}...")
        # Model
        if self.api == "hf":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_ckpt, trust_remote_code=True, device_map="auto"
            ).eval()
        elif self.api == "vllm":
            print(
                f"[LM.py:Txt2TxtGenerator] ==> Loading vllm model {self.model_ckpt} with {torch.cuda.device_count()} GPUs..."
            )
            self.model = LLM(
                model=self.model_ckpt,
                tensor_parallel_size=torch.cuda.device_count(),
                seed=self.vllm_seed,
                gpu_memory_utilization=0.9,
                max_model_len=self.vllm_max_model_len,
            )
        elif self.api == "openai":
            assert self.api_key is not None
            self.client = OpenAI(api_key=self.api_key)
        elif self.api == "anthropic":
            assert self.api_key is not None
            self.client = Anthropic(api_key=self.api_key)
        elif self.api == "together":
            raise NotImplementedError

        # Tokenizer
        if self.api in ["hf", "vllm"]:
            if "OpenELM" in self.model_ckpt:
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
                self.tokenizer.pad_token_id = 0
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt, trust_remote_code=True)
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.tokenizer.padding_side = "left"

    def _format_as_chat_hf(self, input_text: str, system_prompt: str, model: str):
        assert self.is_chat

        if "Qwen" in model:
            """Format using tokenizer's chat template"""
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": input_text}]
            formatted_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif "Phi-3" in model:
            """Format as <|user|>\nQuestion<|end|>\n<|assistant|>"""
            template = "<|user|>\n" "{input_text}<|end|>\n" "<|assistant|>"
            formatted_input = template.format(input_text=input_text)
        elif any(name in model for name in ["OpenELM", "gemma"]):
            formatted_input = input_text
        elif "Sheared-LLaMA" in model:
            template = "You are a helpful assistant. Write a response that appropriately completes the request.\n\n### Input:\n{input_text}\n\n### Response:"
            formatted_input = template.format(input_text=input_text)
        else:
            raise NotImplementedError(f"Hey! Model {model} not supported.")

        return formatted_input

    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 1,
        hf_do_sample: bool = True,
        num_generations=1,
        top_p: float = 1,
        top_k: int = 50,
        repetition_penalty: float = 1,
        temperature: float = 1,
        stop_tokens=None,
        system_prompt: str = "You are a helpful AI assistant.",
    ):
        if self.api == "hf":
            raise NotImplementedError
        elif self.api == "vllm":
            sampling_params = SamplingParams(
                n=num_generations,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                max_tokens=max_new_tokens,
                min_tokens=min_new_tokens,
                stop=stop_tokens,
            )
            vllm_response = self.model.generate(
                [system_prompt + "\n\n" + input_text], sampling_params, use_tqdm=self.vllm_use_tqdm
            )
            if num_generations == 1:
                generated_output = vllm_response[0].outputs[0].text
            else:
                generated_output = [o.text for o in vllm_response[0].outputs]
        elif self.api == "openai":
            assert self.is_chat
            
            if self.model_ckpt in ["o1-mini"]:
                generated_output = []
                for _ in range(num_generations):
                    num_trials = 0
                    while True:
                        try:
                            response = self.client.chat.completions.create(
                                model=self.model_ckpt,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": input_text,
                                    }
                                ],
                            )
                            
                            response_text = response.choices[0].message.content
                            generated_output.append(response_text)
                            break
                        except Exception as e:
                            print(f"[LM.py:Txt2TxtGenerator] ==> OpenAI error: {e}")
                            sleep(1)
                            num_trials += 1
                            if num_trials >= 3:
                                print(f"[LM.py:Txt2TxtGenerator] ==> All API keys failed. Return null string.")
                                generated_output.append("")
                                break
                if num_generations == 1:
                    generated_output = generated_output[0]
            else:
                num_trials = 0
                while True:
                    try:
                        if self.model_ckpt in ["gpt-3.5-turbo-instruct", ]:
                            response = self.client.completions.create(
                                model=self.model_ckpt,
                                prompt=input_text,
                                n=num_generations,
                                stop=stop_tokens,
                                max_tokens=max_new_tokens,
                                top_p=top_p,
                                temperature=temperature,
                            )
                            
                            if num_generations == 1:
                                generated_output = response.choices[0].text
                            else:
                                generated_output = [r.text for r in response.choices]
                        else:
                            response = self.client.chat.completions.create(
                                model=self.model_ckpt,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": input_text},
                                ],
                                n=num_generations,
                                stop=stop_tokens,
                                max_tokens=max_new_tokens,
                                top_p=top_p,
                                temperature=temperature,
                            )

                            if num_generations == 1:
                                generated_output = response.choices[0].message.content
                            else:
                                generated_output = [r.message.content for r in response.choices]

                        break
                    except Exception as e:
                        print(f"[LM.py:Txt2TxtGenerator] ==> OpenAI error: {e}")
                        sleep(1)
                        num_trials += 1
                        if num_trials >= 3:
                            print(f"[LM.py:Txt2TxtGenerator] ==> All API keys failed. Return null string.")
                            generated_output = "" if num_generations == 1 else ["" for _ in range(num_generations)]
                            break
        elif self.api == "anthropic":
            assert self.is_chat
            generated_output = []
            for _ in range(num_generations):
                num_trials = 0
                while True:
                    try:
                        response = self.client.messages.create(
                            model=self.model_ckpt,
                            messages=[
                                {"role": "user", "content": input_text},
                            ],
                            system=system_prompt,
                            max_tokens=max_new_tokens,
                            stop_sequences=stop_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                        )
                        
                        response_text = response.content[0].text
                        generated_output.append(response_text)
                        break
                    except Exception as e:
                        print(f"[LM.py:Txt2TxtGenerator] ==> Anthropic error: {e}")
                        sleep(1)
                        num_trials += 1
                        if num_trials >= 3:
                            print(f"[LM.py:Txt2TxtGenerator] ==> All API keys failed. Return null string.")
                            generated_output.append("")
                            break
            if num_generations == 1:
                generated_output = generated_output[0]
        elif self.api == "together":
            raise NotImplementedError

        return generated_output

    def generate_batch(
        self,
        input_list: list[str],
        max_new_tokens: int = 2048,
        min_new_tokens: int = 1,
        hf_do_sample=True,
        num_generations=1,
        top_p=1,
        top_k=50,
        repetition_penalty=1.1,
        temperature=1,
        stop_tokens=None,
        system_prompt="You are a helpful AI assistant.",
    ):
        raise NotImplementedError

    def calculate_probability_and_perplexity(self, prefix: str, suffix: str):
        assert self.api == "hf"
        assert isinstance(prefix, str) and isinstance(suffix, str)

        # Tokenize the prefix and suffix separately
        tokenized_prefix = self.tokenizer(prefix, return_tensors="pt")
        prefix_ids = tokenized_prefix.input_ids.to("cuda")
        tokenized_suffix = self.tokenizer(suffix, return_tensors="pt", return_special_tokens_mask=True)
        suffix_ids = tokenized_suffix.input_ids.to("cuda")
        special_tokens_mask = tokenized_suffix.special_tokens_mask.to("cuda")
        suffix_ids = suffix_ids[:, special_tokens_mask[0] == 0]

        # Concatenate prefix and suffix token ids
        full_input_ids = torch.cat((prefix_ids, suffix_ids), dim=1)

        """
        # Tokenize the full sequence
        full_sequence = prefix + suffix
        full_sequence_ids = tokenizer(full_sequence, return_tensors='pt').input_ids

        # Check if concatenated token ids match the token ids of the full sequence
        if not torch.equal(full_input_ids, full_sequence_ids):
            raise ValueError("Concatenated prefix and suffix token ids do not match the full sequence token ids")
        """

        # Get logits for the full sequence
        with torch.no_grad():
            outputs = self.model(full_input_ids)
            logits = outputs.logits

        # Get the logits for the tokens following the prefix
        assert logits.ndim == 3
        suffix_logits = logits[:, prefix_ids.size(-1) - 1 : -1, :]

        # Convert logits to probabilities
        probs = F.softmax(suffix_logits, dim=-1)

        # Calculate the probability of the next token sequence
        suffix_probs = torch.gather(probs, -1, suffix_ids.unsqueeze(-1)).squeeze(-1)
        assert suffix_probs.size(-1) > 0

        # Calculate the total probabilities (normalized)
        total_prob = torch.prod(suffix_probs) ** (1 / suffix_probs.size(-1))
        total_prob = total_prob.item()

        # Calculate the sum of negative log probabilities
        negative_log_probs = -torch.log(suffix_probs)
        sum_negative_log_probs = torch.sum(negative_log_probs).item()

        # Calculate the perplexity
        perplexity = math.exp(sum_negative_log_probs / suffix_probs.size(-1))

        def transform_to_list(tensor):
            tensor = tensor.squeeze().tolist()
            if not isinstance(tensor, list):
                tensor = [tensor]

            return tensor

        aux_dict = {
            "suffix_ids": transform_to_list(suffix_ids),
            "suffix_probs": transform_to_list(suffix_probs),
            "suffix_negative_log_probs": transform_to_list(negative_log_probs),
        }
        return total_prob, perplexity, aux_dict


if __name__ == "__main__":
    print("This is a library. Please import it in your script.")
