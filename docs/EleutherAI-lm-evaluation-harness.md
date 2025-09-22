# Language Model Evaluation Harness: Evaluate & Benchmark LLMs

**The Language Model Evaluation Harness is a powerful, versatile framework for rigorously evaluating and benchmarking generative language models across a wide range of tasks. [Visit the original repo](https://github.com/EleutherAI/lm-evaluation-harness) for the latest updates and contributions.**

## Key Features:

*   **Comprehensive Benchmarks:** Supports over 60 standard academic benchmarks with hundreds of subtasks.
*   **Model Compatibility:** Works with models loaded via Hugging Face Transformers, vLLM, SGLang, NeMo, OpenVINO, Triton, and commercial APIs.
*   **Flexible Inference:** Supports fast and memory-efficient inference with vLLM and SGLang, along with options for multi-GPU evaluation, model sharding, and data parallelism.
*   **Reproducibility:** Employs publicly available prompts for consistent and comparable results.
*   **Customization:** Provides easy integration for custom prompts, evaluation metrics, and task creation.
*   **Open LLM Leaderboard Integration:**  Backend for the Hugging Face Open LLM Leaderboard.
*   **Steering Support:**  Evaluate Hugging Face models with steering vectors.

## What's New (v0.4.0 & beyond):

*   **[2025/07]** Added `think_end_token` arg to `hf` (token/str), `vllm` and `sglang` (str) for stripping CoT reasoning traces from models that support it.
*   **[2025/03]**  Added support for steering HF models!
*   **[2025/02]** Added [SGLang](https://docs.sglang.ai/) support!
*   **[2024/09]** Prototype multimodal evaluation with `hf-multimodal`, `vllm-vlm`, and `mmmu` task.
*   **[2024/07]** Refactored and updated API model support with batched and async requests, including local model API.
*   **New Open LLM Leaderboard tasks have been added !**

## Installation:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

Explore optional dependencies for expanded functionality using the table in the [original README](https://github.com/EleutherAI/lm-evaluation-harness#optional-extras).

## Getting Started:

See [user guide](docs/interface.md) and [task descriptions](./lm_eval/tasks/README.md) for full functionality.

### Evaluate a Hugging Face Model:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

For multi-GPU evaluation with Hugging Face `accelerate`, using the  `accelerate launch` command is recommended (See original README for details).

## Supported Models & APIs:

This library supports numerous model types, including:

*   Hugging Face Transformers
*   vLLM
*   SGLang
*   NVIDIA NeMo
*   OpenVINO
*   Llama.cpp
*   OpenAI, Anthropic, Textsynth, and local API servers.

See the original README for detailed instructions and examples for each model type and API.

## Advanced Usage & Tips:

*   **GGUF Models:** Evaluate GGUF models by specifying the model path, the GGUF file, and a tokenizer.
*   **Steering:**  Apply steering vectors to Hugging Face models using PyTorch files or CSV configurations.
*   **Caching:** Use `--use_cache <DIR>` to accelerate evaluation by skipping previously evaluated samples.
*   **Saving & Logging:**  Use `--output_path` to save results and `--log_samples` to log model responses.
*   **Visualization:** Integrate with Weights & Biases (W&B) and Zeno for visualizing and analyzing your results.

## Contributing & Support:

Check out the [open issues](https://github.com/EleutherAI/lm-evaluation-harness/issues) and join the [EleutherAI Discord](https://discord.gg/eleutherai) for support and collaboration.