<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png" alt="lmms-eval Logo">
</p>

# LMMs-Eval: The Comprehensive Evaluation Suite for Large Multimodal Models 

**Quickly and efficiently evaluate Large Multimodal Models (LMMs) with LMMs-Eval, a powerful and versatile framework.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> **Accelerate LMM Development:** `lmms-eval` supports a wide range of text, image, video, and audio tasks, providing a robust and flexible evaluation framework.

🌐 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features of LMMs-Eval

*   **Extensive Task Coverage:** Evaluate LMMs across diverse modalities, including text, images, videos, and audio.
*   **Broad Model Support:** Compatible with over 30 different LMMs, including many of the most popular models.
*   **Efficient Evaluation:** Integrated with `vLLM` and `openai_compatible` for accelerated inference and API-based model support.
*   **Reproducible Results:** Detailed results and environment information provided to facilitate reproducibility.
*   **Community-Driven:** Active development with contributions from the community, constantly incorporating new tasks and models.
*   **Easy to Use:** Simple installation and clear usage examples for seamless integration into your workflow.

## What's New

*   **Major Updates:** 🚀 The recent `lmms-eval-0.4` release introduces significant features and improvements. See the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **Audio Evaluation Support:** 🔈 Now supports audio evaluations for audio models like Qwen2-Audio and Gemini-Audio across tasks such as AIR-Bench, Clotho-AQA, LibriSpeech, and more.
*   **New Tasks Supported:** 🎉 Added support for [PhyX](https://phyx-bench.github.io/), [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), and other benchmarks.

---

## Installation

Install LMMs-Eval with the following steps:

**For direct usage:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

**For development:**

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Check the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5's paper results. Variations may occur due to torch/cuda versions; [results check](miscs/llava_result_check.md) is provided.

</details>

**Dependencies:**

*   If you're using caption datasets (`coco`, `refcoco`, `nocaps`), you'll need `java==1.8.0` to use the pycocoeval API:

```bash
conda install openjdk=8
```

*   Then, verify your Java version: `java -version`

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

We provide detailed evaluation results of the LLaVA series models.

Access the Google Sheet with detailed results [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing).

We provide the raw data exported from Weights & Biases for the detailed results of the LLaVA series models on different datasets. You can access the raw data [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

<p align="center" width="100%">
<img src="https://i.postimg.com/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

</details>
<br>

*   To test [VILA](https://github.com/NVlabs/VILA), you must install the following:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

## Usage Examples

> Find more examples in [examples/models](examples/models)

**Evaluate OpenAI-Compatible Models:**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluate with vLLM:**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Evaluate LLaVA-OneVision:**

```bash
bash examples/models/llava_onevision.sh
```

**Evaluate LLaMA-3.2-Vision:**

```bash
bash examples/models/llama_vision.sh
```

**Evaluate Qwen2-VL:**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Evaluate LLaVA on MME:**

To test LLaVA 1.5, clone the repo from [LLaVA](https://github.com/haotian-liu/LLaVA) and run:

```bash
bash examples/models/llava_next.sh
```

**Evaluate with Tensor Parallelism (for larger models like llava-next-72b):**

```bash
bash examples/models/tensor_parallel.sh
```

**Evaluate with SGLang (for larger models):**

```bash
bash examples/models/sglang.sh
```

**Evaluate with vLLM (for larger models):**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Additional Parameters:**

```bash
python3 -m lmms_eval --help
```

**Environmental Variables:**

Set these environment variables before running your evaluations:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include 
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Troubleshooting Common Issues:**

Address potential errors related to `httpx` and `protobuf`:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
python3 -m pip install numpy==1.26; # If you use numpy==2.x, it might cause errors
python3 -m pip install sentencepiece; # Required sometimes
```

## Add Your Own Model and Dataset

Refer to our [documentation](docs/README.md).

## Acknowledgements

`lmms-eval` is inspired by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Consult the [lm-evaluation-harness docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for relevant info.

---

### Changes from the Original API:

*   Build context now passes only `idx` and processes images/docs during model response phase.
*   `Instance.args` (lmms_eval/api/instance.py) now holds a list of images.
*   Model class creation is required for each LMM due to differing input/output formats, which will be addressed in the future.

---

## Citations

```shell
@misc{zhang2024lmmsevalrealitycheckevaluation,
      title={LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models}, 
      author={Kaichen Zhang and Bo Li and Peiyuan Zhang and Fanyi Pu and Joshua Adrian Cahyono and Kairui Hu and Shuai Liu and Yuanhan Zhang and Jingkang Yang and Chunyuan Li and Ziwei Liu},
      year={2024},
      eprint={2407.12772},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.12772}, 
}

@misc{lmms_eval2024,
    title={LMMs-Eval: Accelerating the Development of Large Multimoal Models},
    url={https://github.com/EvolvingLMMs-Lab/lmms-eval},
    author={Bo Li*, Peiyuan Zhang*, Kaichen Zhang*, Fanyi Pu*, Xinrun Du, Yuhao Dong, Haotian Liu, Yuanhan Zhang, Ge Zhang, Chunyuan Li and Ziwei Liu},
    publisher    = {Zenodo},
    version      = {v0.1.0},
    month={March},
    year={2024}
}