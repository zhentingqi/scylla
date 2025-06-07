# SCYLLA
Code repo for paper [Quantifying Generalization Complexity for Large Language Models](https://arxiv.org/abs/2410.01769) (ICLR 2025).

## Intro
While large language models (LLMs) have shown exceptional capabilities in understanding complex queries and performing sophisticated tasks, their generalization abilities are often deeply entangled with memorization, necessitating more precise evaluation. To address this challenge, we introduce Scylla, a dynamic evaluation framework that quantitatively measures the generalization abilities of LLMs. Scylla disentangles generalization from memorization via assessing model performance on both in-distribution (ID) and out-of-distribution (OOD) data through 20 tasks across 5 levels of complexity. Through extensive experiments, we uncover a non-monotonic relationship between task complexity and the performance gap between ID and OOD data, which we term the generalization valley. Specifically, this phenomenon reveals a critical threshold - referred to as critical complexity - where reliance on non-generalizable behavior peaks, indicating the upper bound of LLMs' generalization capabilities. As model size increases, the critical complexity shifts toward higher levels of task complexity, suggesting that larger models can handle more complex reasoning tasks before over-relying on memorization. Leveraging Scylla and the concept of critical complexity, we benchmark 28 LLMs including both open-sourced models such as LLaMA and Qwen families, and close-sourced models like Claude and GPT, providing a more robust evaluation and establishing a clearer understanding of LLMs' generalization capabilities.

## Prerequisite
- CUDA 12.4
- `torch==2.3.0`
- `transformers==4.44.2`
- `vllm==0.5.1`

## Usage
We provide an example of running the entire pipeline:
```bash
bash jobs/run.sh
```

## Citation
If you find our work helpful, please consider citing it!
```txt
@article{qi2024quantifying,
  title={Quantifying Generalization Complexity for Large Language Models},
  author={Qi, Zhenting and Luo, Hongyin and Huang, Xuliang and Zhao, Zhuokai and Jiang, Yibo and Fan, Xiangjun and Lakkaraju, Himabindu and Glass, James},
  journal={arXiv preprint arXiv:2410.01769},
  year={2024}
}
```
