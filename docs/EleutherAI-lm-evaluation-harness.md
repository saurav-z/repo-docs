# Language Model Evaluation Harness: Evaluate & Benchmark LLMs (with link to EleutherAI/lm-evaluation-harness)

**Unleash the power of rigorous LLM evaluation!** The Language Model Evaluation Harness ([EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)) is a comprehensive framework for benchmarking generative language models, providing a unified approach to testing and comparing model performance across a wide range of tasks and architectures.

**Key Features:**

*   **Extensive Benchmark Suite:** Evaluate your LLMs on over 60 standard academic benchmarks, including hundreds of subtasks and variants, ensuring thorough performance assessment.
*   **Flexible Model Support:** Seamlessly integrate models from Hugging Face Transformers (including quantization), GPT-NeoX, Megatron-DeepSpeed, vLLM, and commercial APIs (OpenAI, Anthropic, TextSynth, Watsonx.ai).
*   **High-Performance Inference:** Leverage vLLM and SGLang for faster inference, especially when using multiple GPUs.
*   **Reproducibility and Comparability:** Utilize publicly available prompts and evaluation metrics to ensure that your results can be easily compared to those of other research.
*   **Customization & Extensibility:** Easily create custom prompts, evaluation metrics, and tasks to tailor the framework to your specific needs.
*   **Steering Vector Support**: Implement steering vectors for improved performance with HF models.
*   **Multi-GPU Evaluation**: Implement data-parallel, tensor-parallel, and pipeline-parallel evaluation, depending on the model's needs.
*   **Result Visualization:** Easily visualize and analyze results using both Weights & Biases (W&B) and Zeno.

**Why Use the Language Model Evaluation Harness?**

The Language Model Evaluation Harness is the backbone of Hugging Face's [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) and is used internally by numerous organizations, including NVIDIA and Cohere. By providing a standardized and extensible framework, this tool empowers researchers, developers, and organizations to objectively assess and improve the performance of language models.

**New in v0.4.0 (Latest Updates):**

*   Added `think_end_token` arg to `hf` (token/str), `vllm` and `sglang` (str) for stripping CoT reasoning traces.
*   Added support for steering HF models!
*   Added [SGLang](https://docs.sglang.ai/) support!
*   Improved multimodal input (text+image), text output tasks, and added `hf-multimodal` and `vllm-vlm` model types.
*   API model support refactored with batched and async requests.
*   New Open LLM Leaderboard tasks added.

**Installation:**

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

**Get Started:**

Explore the comprehensive [user guide](docs/interface.md) and the command-line interface with `lm-eval -h` for detailed usage instructions.
A list of supported tasks can be viewed with `lm-eval --tasks list`. Task descriptions and links to corresponding subfolders are provided [here](./lm_eval/tasks/README.md).

**Access the full list of models here:** [https://github.com/EleutherAI/lm-evaluation-harness#model-apis-and-inference-servers](https://github.com/EleutherAI/lm-evaluation-harness#model-apis-and-inference-servers)

**Further Exploration:**

*   [Documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs)
*   [Open Issues](https://github.com/EleutherAI/lm-evaluation-harness/issues)
*   [EleutherAI Discord server](https://discord.gg/eleutherai)

**Cite Us:**

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