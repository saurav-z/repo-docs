# Language Model Evaluation Harness: Evaluate LLMs with Ease

**Unleash the power of comprehensive language model evaluation with the Language Model Evaluation Harness, a versatile and widely-used framework. [[Original Repo](https://github.com/EleutherAI/lm-evaluation-harness)]**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10256836.svg)](https://doi.org/10.5281/zenodo.10256836)

## Key Features

*   **Extensive Benchmarks:** Access over 60 standard academic benchmarks with hundreds of subtasks.
*   **Model Flexibility:** Supports models from Hugging Face Transformers, vLLM, SGLang, NeMo, commercial APIs (OpenAI, Anthropic, TextSynth), and local/self-hosted servers.
*   **Fast Inference:** Optimize evaluation with vLLM, SGLang and other efficient inference backends.
*   **Reproducibility:**  Leverage publicly available prompts for consistent and comparable results.
*   **Customization:**  Easily integrate custom prompts, evaluation metrics, and model configurations.
*   **Multi-GPU Support:** Utilize multi-GPU evaluation via `accelerate` for faster results.
*   **Result Visualization:** Visualize results using Weights & Biases (W&B) and Zeno for deeper insights.
*   **Steered Models:** Evaluate Hugging Face models with steering vectors via PyTorch files or CSV files.

## What's New

*   **[2025/07]** Added `think_end_token` arg to `hf` (token/str), `vllm` and `sglang` (str) for stripping CoT reasoning traces from models that support it.
*   **[2025/03]** Added support for steering HF models!
*   **[2025/02]** Added [SGLang](https://docs.sglang.ai/) support!
*   **[2024/09]** Prototyping text+image multimodal input, text output tasks, and the `hf-multimodal` and `vllm-vlm` model types and `mmmu` task.
*   **[2024/07]** Refactored and updated API model support, including batched and async requests.
*   **[2024/07]** Added new Open LLM Leaderboard tasks!
*   **v0.4.0 release available**: includes new Open LLM Leaderboard tasks, internal refactoring, config-based task creation, prompt design improvements, and new modeling libraries (vLLM, MPS, and more).

## Installation

1.  Clone the repository:
    ```bash
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    ```
2.  Install the package:
    ```bash
    pip install -e .
    ```
3.  Install extras for extended functionality:  `pip install -e ".[NAME]"`.

## Basic Usage

### Hugging Face Transformers Example

Evaluate a model on a task using the Hugging Face backend:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

### Advanced Usage Tips

*   **GGUF Models:** Evaluate GGUF models by specifying the path to the model weights and tokenizer.
*   **Multi-GPU with `accelerate`:** Utilize data and/or model parallelism for faster evaluation.
*   **vLLM:**  Leverage vLLM for optimized inference and continuous batching.
*   **SGLang:** Use SGLang for efficient offline batch inference.
*   **API Support:** Evaluate models through various commercial and local APIs.
*   **Steered Models:** Evaluate Hugging Face models with steering vectors.
*   **Saving & Caching:** Save and cache results using `--output_path` and `--use_cache`.

## Contributing

We welcome contributions!  Check out our [open issues](https://github.com/EleutherAI/lm-evaluation-harness/issues) and refer to the [documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for more details.  Join the [EleutherAI Discord](https://discord.gg/eleutherai) for support and discussion.

## Cite Us

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