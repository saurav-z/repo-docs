# Language Model Evaluation Harness: Evaluate Your LLMs with Ease

**Quickly and reliably benchmark your Large Language Models (LLMs) using the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) – a comprehensive framework trusted by leading organizations.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10256836.svg)](https://doi.org/10.5281/zenodo.10256836)

## Key Features

*   **Extensive Benchmark Support:** Evaluate LLMs on 60+ standard academic benchmarks, encompassing hundreds of subtasks and variations.
*   **Flexible Model Integration:** Supports models from Hugging Face `transformers` (including quantization), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/), and commercial APIs (OpenAI, TextSynth, and more).
*   **Optimized Inference:** Leverage fast and memory-efficient inference with [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://docs.sglang.ai/).
*   **Reproducible Results:** Evaluate models with publicly available prompts, ensuring reproducibility and comparability.
*   **Customization & Extensibility:** Easily integrate custom prompts, evaluation metrics, and support for adapters (e.g., LoRA) and local models.
*   **Multi-GPU & Multi-Node Support:** Efficiently evaluate large models with data parallelism and model sharding via `accelerate`.
*   **Integration with Hugging Face Hub:** Seamlessly log results and samples to the Hugging Face Hub for sharing and collaboration.
*   **Visualization Tools:** Integrate with [Weights & Biases](https://wandb.ai/site) (W&B) and [Zeno](https://zenoml.com) for in-depth analysis and visualization.

## What's New? (Recent Updates)

*   **2025/07:** Added `think_end_token` argument for removing CoT traces.
*   **2025/03:** Added support for steering HF models.
*   **2025/02:** Added [SGLang](https://docs.sglang.ai/) support.
*   **2024/09:** Prototyping multimodal input and output tasks; added `hf-multimodal` and `vllm-vlm` model types.
*   **2024/07:** Refactored and improved API model support with batched and async requests.  Added new Open LLM Leaderboard tasks.
*   **v0.4.0 Release Highlights:** New Open LLM Leaderboard tasks, config-based task creation, Jinja2 prompt support, speedups, new modeling libraries (vLLM, MPS), and more.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    ```

2.  **Install the package:**

    ```bash
    pip install -e .
    ```

3.  **Install optional dependencies:** Refer to the table below for the extras you need.

## Basic Usage

### Hugging Face Transformers Example

Evaluate a model from the Hugging Face Hub (e.g., GPT-J-6B) on the `hellaswag` task:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

### Other Frameworks

Refer to the documentation for how to run a specific model (NVIDIA NeMo, vLLM, SGLang, and others).

## Advanced Use Cases

*   **GGUF Model Evaluation:** Pass the path to the GGUF model and tokenizer using `--model_args`.
*   **Multi-GPU Evaluation:** Utilize the `accelerate` library for data-parallel and model-sharded evaluation.
*   **Steered Models:** Evaluate models with steering vectors using the `--model steered` option.
*   **API Model Integration:** Evaluate models through the OpenAI API, TextSynth, or local inference servers using the corresponding `--model` arguments and API keys.
*   **Result Caching and Logging:** Save results using `--output_path` and `--use_cache`, and log samples with `--log_samples`.
*   **Visualize results:** Use the Zeno and W&B integrations.

## Optional Extras

Install optional dependencies using `pip install -e ".[NAME]"`

| NAME                 | Description                    |
|----------------------|--------------------------------|
| tasks                | All task-specific dependencies |
| api                  | API models (Anthropic, OpenAI, local) |
| acpbench             | ACP Bench tasks                |
| audiolm_qwen   | Qwen2 audio models                    |
| ifeval               | IFEval task                    |
| japanese_leaderboard | Japanese LLM tasks             |
| gptq           | AutoGPTQ models                       |
| longbench            | LongBench tasks                |
| gptqmodel      | GPTQModel models                       |
| math                 | Math answer checking           |
| hf_transfer    | Speed up HF downloads                 |
| multilingual         | Multilingual tokenizers        |
| ibm_watsonx_ai | IBM watsonx.ai models                 |
| ruler                | RULER tasks                    |
| ipex           | Intel IPEX backend                    |
|                      |                                |
| dev                  | Linting & contributions        |
| mamba          | Mamba SSM models                      |
| promptsource         | PromptSource prompts           |
| neuronx        | AWS inf2 instances                    |
| sentencepiece        | Sentencepiece tokenizer        |
| optimum        | Intel OpenVINO models                 |
| testing              | Run test suite                 |
| sae_lens       | SAELens model steering                |
| unitxt               | Run unitxt tasks               |
| wandb                | Weights & Biases               |
| sparsify       | Sparsify model steering               |
| zeno                 | Result visualization           |
| vllm           | vLLM models                           |

## Contributing

Contributions are welcome! Check out the [open issues](https://github.com/EleutherAI/lm-evaluation-harness/issues) and the [documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for guidance.

## Support

For support, open an issue on GitHub or join the [EleutherAI Discord server](https://discord.gg/eleutherai).

## Cite

```text
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {The Language Model Evaluation Harness},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}
```