from utils.task_utils import get_elements_from_example, format_element_or_example

import os, json, re
from time import time, sleep


gen = None  # could be a GPT object


TEMPLATES = {
    "list": 'Below is a text sequence. I need you to identify all the lists and nested lists. A list is a sequence of items separated by ",", " ", or ", "  and must be enclosed by square brackets, and each element must be either a number or a string. For example, [1, 2, 3], ["foo", "bar", "baz"] are valid lists, and [[1,2], [3,4]] is a valid nested list. Please find all the valid lists or nested lists in the given text, and output them as a JSON list object. Your output must start with "[" and end with "]".\n\nThe text sequence:\n%TEXT%\n\nYour JSON output:\n',
    "string_dict": 'Below is a text sequence. I need you to identify all the dictionaries with the following format: {"input": <str>}. For example, {"input": "abc"} is a valid dictionary. Please find all the valid dictionaries in the given text, and output them as a JSON list object as a list of dictionaries. Your output must start with "[" and end with "]", and each dictionary element in the list must start with "{" and end with "}", and the key of the dictionary must be "input" while the value must be a string.\n\nThe text sequence:\n%TEXT%\n\nYour JSON output:\n',
    "string_list_dict": 'Below is a text sequence. I need you to identify all the dictionaries with the following format: {"input": [<str>, <str>]}. For example, {"input": ["abc", "def"]} is a valid dictionary. Please find all the valid dictionaries in the given text, and output them as a JSON list object as a list of dictionaries. Your output must start with "[" and end with "]", and each dictionary element in the list must start with "{" and end with "}", and the key of the dictionary must be "input" while the value must be a list of exactly two strings.\n\nThe text sequence:\n%TEXT%\n\nYour JSON output:\n',
}


def _template_formatter(template, text):
    return template.replace("%TEXT%", text)


POST_PROCESSER = {"list": lambda x: x, "string_dict": lambda x: x["input"], "string_list_dict": lambda x: x["input"]}


def _gpt_extract(text: str, type: str):
    assert type in TEMPLATES, f"Invalid type: {type}"
    formatted_template = _template_formatter(TEMPLATES[type], text)
    trials = 0
    list_of_lists = None
    while trials < 2:
        try:
            output = gen.generate(formatted_template, max_new_tokens=256)
            output = output.replace("```json", "").replace("```", "")
            output = output.split("\n\n")[0]
            list_of_lists = json.loads(output)
            break
        except Exception as e:
            trials += 1
            continue
    return list_of_lists


def _re_extract(text: str, type: str):
    def extract_lists(text):
        # Matches valid lists recursively (captures nested lists)
        def find_lists(text):
            bracket_stack = []
            start_idx = -1
            result = []
            for i, char in enumerate(text):
                if char == '[':
                    if len(bracket_stack) == 0:
                        start_idx = i
                    bracket_stack.append(char)
                elif char == ']':
                    if bracket_stack:  # Ensure that we only pop if there is something to pop
                        bracket_stack.pop()
                        if len(bracket_stack) == 0 and start_idx != -1:
                            result.append(text[start_idx:i+1])
                            start_idx = -1
            return result
        
        matches = find_lists(text)
        lists = []
        for match in matches:
            try:
                parsed_list = json.loads(match)
                if isinstance(parsed_list, list):
                    lists.append(parsed_list)
            except json.JSONDecodeError:
                continue
        return lists
    
    def extract_string_dicts(text):
        dict_pattern = r'{"input":\s*"(.*?)"}'
        matches = re.findall(dict_pattern, text)
        dicts = [{"input": match} for match in matches]
        return dicts
    
    def extract_string_list_dicts(text):
        # Ensure that the values inside the lists are strictly strings
        dict_pattern = r'{"input":\s*\["([^"]+)",\s*"([^"]+)"\]}'
        matches = re.findall(dict_pattern, text)
        dicts = [{"input": [match[0], match[1]]} for match in matches]
        return dicts
    
    if type == "list":
        return extract_lists(text)
    elif type == "string_dict":
        return extract_string_dicts(text)
    elif type == "string_list_dict":
        return extract_string_list_dicts(text)
    else:
        raise ValueError(f"Invalid type: {type}")


def extract_elements_and_examples(model_output: str, task):
    """Extract the most fine-grained individual items in a test sample

    Args:
        model_output (str): The model's output

    Returns:
        List[str]: A list of the most fine-grained individual items in a test sample (un-deduped)
    """
    assert isinstance(model_output, str)

    # examples = _gpt_extract(model_output, task.example_type_for_gpt_extract)
    examples = _re_extract(model_output, task.example_type_for_gpt_extract)
    
    if examples is None:
        return [], []

    all_valid_elements = []
    all_valid_examples = []

    for i, example in enumerate(examples):
        try:
            example = POST_PROCESSER[task.example_type_for_gpt_extract](example)
        except:
            # print(f"Invalid example: {example}")
            continue

        elements = get_elements_from_example(example, task)
        all_valid_elements.extend(elements)
        
        # print(f"==> Checking example {i}: {example}")
        valid = task.check_example(example)
        if valid:
            all_valid_examples.append(example)
            # print(f"Valid example")
        else:
            try:
                # print(f"Invalid example, trying to transform...")
                example = task.try_transform_invalid_example(example)
                all_valid_examples.append(example)
                # print(f"Transformed example: {example}")
            except:
                # print(f"Invalid example, failed to transform")
                continue
        
    # print("==> Formatting elements...")
    processed_elements = [format_element_or_example(e, task) for e in all_valid_elements]
    # print("==> Formatting examples...")
    processed_examples = [format_element_or_example(e, task) for e in all_valid_examples]
    
    assert all([task.check_element(e) for e in processed_elements])
    assert all([task.check_example(e) for e in processed_examples])
    
    # print(f"==> Extraction done.")

    return processed_elements, processed_examples
