# Language Model Evaluation Harness: The Premier Framework for LLM Benchmarking ([EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness))

**Unleash the power of comprehensive LLM evaluation with the Language Model Evaluation Harness, the leading tool for assessing and comparing generative language models.**

---

## Key Features:

*   **Extensive Benchmark Support:** Evaluate models on over 60 standard academic benchmarks, including hundreds of subtasks and variants.
*   **Flexible Model Integration:** Supports models from [Hugging Face Transformers](https://github.com/huggingface/transformers/), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://docs.sglang.ai/), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/) , and commercial APIs ([OpenAI](https://openai.com), [Anthropic](https://www.anthropic.com), [TextSynth](https://textsynth.com/), and more).
*   **Advanced Features:** Includes support for adapters (e.g., LoRA), custom prompts and evaluation metrics, output post-processing, answer extraction, and multiple LM generations.
*   **Reproducible and Comparable Results:** Utilizes publicly available prompts and evaluation metrics for consistent results.
*   **Multi-GPU & Accelerated Inference:** Supports multi-GPU evaluation with Hugging Face `accelerate`, vLLM, and SGLang for faster and more efficient assessment.
*   **Integrated Visualization:** Compatible with Weights & Biases and Zeno for in-depth analysis and visualization of results.
*   **Steered Model Support:** Evaluate Hugging Face `transformers` models with steering vectors, improving model performance.
*   **NVIDIA NeMo and OpenVINO Model Support:** Evaluation of NVIDIA NeMo and OpenVINO models using the respective backends.

## What's New:

*   **[2024/07]** New Open LLM Leaderboard tasks have been added !
*   **[2024/07]** Refactored and updated API model support.
*   **[2024/07]** Added `think_end_token` arg to strip CoT reasoning traces.
*   **[2025/03]** Added steering support for HF models!
*   **[2025/02]** Added [SGLang](https://docs.sglang.ai/) support!
*   **[2024/09]** Prototyping text+image multimodal input and text output tasks.

## Installation:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

For optional dependencies and advanced features, consult the [Optional Extras](#optional-extras) section below.

## Basic Usage:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

For detailed usage instructions, see the [User Guide](./docs/interface.md). Explore supported tasks with `lm-eval --tasks list`. Task descriptions are available [here](./lm_eval/tasks/README.md).

## Advanced Usage:

*   **Hugging Face Transformers:** Supports models hosted on the Hugging Face Hub, local models, and models loaded via `transformers`. Includes support for revision control, data types, and batch size automation.  See [Hugging Face transformers](#hugging-face-transformers) for details.
*   **GGUF Model Evaluation:** Evaluate models in GGUF format.  See [Evaluating GGUF Models](#evaluating-gguf-models)
*   **Multi-GPU Evaluation:** Leverages Hugging Face `accelerate` for data-parallel and model sharding evaluations.  See [Multi-GPU Evaluation with Hugging Face `accelerate`](#multi-gpu-evaluation-with-hugging-face-accelerate)
*   **Steered Hugging Face Models:** Evaluate models with steering vectors. See [Steered Hugging Face `transformers` models](#steered-hugging-face-transformers-models)
*   **NVIDIA NeMo:** Support for evaluating NVIDIA NeMo models, including multi-GPU options.  See [NVIDIA `nemo` models](#nvidia-nemo-models)
*   **vLLM:** Integrate with vLLM for accelerated inference. See [Tensor + Data Parallel and Optimized Inference with `vLLM`](#tensor--data-parallel-and-optimized-inference-with-vllm)
*   **SGLang:** Support for efficient offline batch inference. See [Tensor + Data Parallel and Fast Offline Batching Inference with `SGLang`](#tensor--data-parallel-and-fast-offline-batching-inference-with-sglang)
*   **Model APIs:**  Supports various API models, including OpenAI, Anthropic, and TextSynth. See [Model APIs and Inference Servers](#model-apis-and-inference-servers)

## Saving & Caching Results:

*   Use `--output_path` to save results.
*   Use `--log_samples` to log model responses.
*   Cache results using `--use_cache <DIR>` to skip previously evaluated samples.
*   Push results and samples to the Hugging Face Hub with the `--hf_hub_log_args` flag.

## Visualizing Results:

*   Integrate with [Zeno](https://zenoml.com) and [Weights & Biases](https://wandb.ai/site) for comprehensive visualization and analysis.

## Contributing:

Contributions are welcome! See the [open issues](https://github.com/EleutherAI/lm-evaluation-harness/issues), [documentation pages](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs), and [Contributing](#contributing) section for details.

## Optional Extras:

[See table in the original README]

## Cite as:

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