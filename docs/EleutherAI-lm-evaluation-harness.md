# Language Model Evaluation Harness: Evaluate and Compare LLMs with Ease

**Unleash the power of comprehensive LLM evaluation:** The Language Model Evaluation Harness (**[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**) provides a unified, flexible, and reproducible framework for assessing the performance of generative language models on a wide range of benchmarks.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10256836.svg)](https://doi.org/10.5281/zenodo.10256836)

## Key Features

*   **Extensive Benchmarks:** Evaluate on 60+ standard academic benchmarks with hundreds of subtasks and variants, covering a wide spectrum of LLM capabilities.
*   **Model Flexibility:** Supports models from Hugging Face Transformers (including quantization with GPTQModel and AutoGPTQ), GPT-NeoX, and Megatron-DeepSpeed, with a tokenization-agnostic interface.
*   **Fast Inference:** Leverage vLLM, SGLang, and other optimized backends for efficient inference, including support for tensor/data parallelism and continuous batching.
*   **API Integration:** Seamlessly evaluate models via commercial APIs (OpenAI, Anthropic, TextSynth, Watsonx.ai) and local inference servers (OpenAI-compatible).
*   **Reproducibility:** Evaluate with publicly available prompts, ensuring consistent and comparable results across studies.
*   **Customization:** Easily integrate custom prompts, evaluation metrics, and task configurations.
*   **Advanced Capabilities:** Includes support for adapters (e.g., LoRA), steered models, multi-GPU evaluation with accelerate, and various model formats.
*   **Visualization & Analysis:** Integrates with Weights & Biases (W&B) and Zeno for comprehensive result visualization and analysis.

## Latest Updates

*   **(2025/07)** Added `think_end_token` arg to `hf`, `vllm` and `sglang`
*   **(2025/03)** Support for Steering HF models.
*   **(2025/02)** Added [SGLang](https://docs.sglang.ai/) support!
*   **(2024/09)** Prototyping multimodal input (text+image), text output tasks with `hf-multimodal`, `vllm-vlm` and `mmmu` task.
*   **(2024/07)** Refactored and improved API model support, with batched and async requests and added Open LLM Leaderboard tasks.

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

3.  **Install optional dependencies:**
    *   Install extras for specific model types or features as needed. E.g., install vLLM: `pip install -e ".[vllm]"` See "Optional Extras" section below.

## Basic Usage

### Evaluating a Hugging Face Model

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

**Important:** Remember to specify the correct device (`--device`) and adjust the `--batch_size` for optimal performance. See the original [README](https://github.com/EleutherAI/lm-evaluation-harness) for more options and details.

### Key Model Types and Backends

*   **`hf`:** Hugging Face Transformers (supports quantization, adapters, and more).
*   **`vllm`:** For fast inference with vLLM (install with `pip install -e ".[vllm]"`).
*   **`sglang`:** Efficient offline batch inference (install as per [SGLang instructions](https://docs.sglang.ai/start/install.html#install-sglang)).
*   **`openai-completions`, `openai-chat-completions`, `anthropic`, `anthropic-chat`, `textsynth`, `local-completions`, `local-chat-completions`:** Utilize various API and server-based models.
*   **`nemo_lm`:** Evaluate NVIDIA NeMo models.

### Advanced Tips

*   Use `--model_args` to pass arguments to model constructors (e.g., `pretrained=/path/to/model`, `revision=step100000`, etc.).
*   Use `--batch_size auto` for automated batch size detection with vLLM and some other backends.
*   Leverage `--use_cache <DIR>` to cache results and resume runs.
*   Use `--log_samples` to save model responses for analysis.
*   Push results to the Hugging Face Hub with `--hf_hub_log_args`.

## Optional Extras

Install extras for enhanced functionality:

| Name             | Description                     | Example                                          |
| ---------------- | ------------------------------- | ------------------------------------------------ |
| `tasks`          | Task-specific dependencies      | `pip install -e ".[tasks]"`                      |
| `vllm`           | vLLM backend                    | `pip install -e ".[vllm]"`                       |
| `sglang`         | SGLang backend                  | (Follow SGLang installation instructions)         |
| `wandb`          | Weights & Biases integration    | `pip install -e ".[wandb]"`                      |
| `zeno`           | Zeno visualization              | `pip install -e ".[zeno]"`                       |
| `gptq`           | AutoGPTQ models                 | `pip install -e ".[gptq]"`                       |
| ... (See full list in the original README) | Various other integrations and dependencies | ...                                             |

## Contributing

We welcome contributions! Check out our [open issues](https://github.com/EleutherAI/lm-evaluation-harness/issues) and the [documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for more information.

## Cite

```
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