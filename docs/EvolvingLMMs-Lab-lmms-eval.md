<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: The Comprehensive Evaluation Suite for Large Multimodal Models

**Quickly and effectively benchmark your LMMs with LMMs-Eval, the leading evaluation framework for text, image, video, and audio tasks.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> LMMs-Eval provides a robust and efficient framework for evaluating Large Multimodal Models (LMMs), supporting a wide array of tasks across text, image, video, and audio modalities.  [Visit our GitHub Repository](https://github.com/EvolvingLMMs-Lab/lmms-eval) to get started!

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Comprehensive Task Support:** Evaluate your LMMs across a vast range of text, image, video, and audio tasks.
*   **Extensive Model Compatibility:**  Supports a growing list of over 30 LMMs, including popular architectures.
*   **Efficient Evaluation Framework:**  Designed for speed and consistency, streamlining the evaluation process.
*   **Modular Design:** Easily add custom models and datasets to tailor the evaluation to your specific needs.
*   **Reproducibility:**  Provides scripts and resources to reproduce results from major LMM papers.
*   **Active Development:**  Benefit from continuous updates, new features, and community contributions.
*   **API Support:** Evaluate models with OpenAI, vLLM, and SGLang.

## Announcements

*   **[2025-07]** 🚀🚀 `lmms-eval-0.4` has been released with major updates and improvements. See the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
*   **[2025-04]** 🚀🚀 Introducing [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/) — a compact yet mighty audio model. Officially supports batched evaluations!
*   **[2025-07]** 🎉🎉 New task [PhyX](https://phyx-bench.github.io/) added!
*   **[2025-06]** 🎉🎉 New task [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA) added!
*   **[2025-02]** 🚀🚀 Integrated `vllm` and `openai_compatible`.

<details>
<summary>Below is a chronological list of recent tasks, models, and features added by our amazing contributors. </summary>

- [2025-01] 🎓🎓 We have released our new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826). Please refer to the [project page](https://videommmu.github.io/) for more details.
- [2024-12] 🎉🎉 We have presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296), jointly with [MME Team](https://github.com/BradyFU/Video-MME) and [OpenCompass Team](https://github.com/open-compass).
- [2024-11] 🔈🔊 The `lmms-eval/v0.3.0` has been upgraded to support audio evaluations for audio models like Qwen2-Audio and Gemini-Audio across tasks such as AIR-Bench, Clotho-AQA, LibriSpeech, and more. Please refer to the [blog](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.3.md) for more details!
- [2024-10] 🎉🎉 We welcome the new task [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), a vision-centric VQA benchmark (NeurIPS'24) that challenges vision-language models with simple questions about natural imagery.
- [2024-10] 🎉🎉 We welcome the new task [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) for fine-grained temporal understanding and reasoning for videos, which reveals a huge (>30%) human-AI gap.
- [2024-10] 🎉🎉 We welcome the new tasks [VDC](https://rese1f.github.io/aurora-web/) for video detailed captioning, [MovieChat-1K](https://rese1f.github.io/MovieChat/) for long-form video understanding, and [Vinoground](https://vinoground.github.io/), a temporal counterfactual LMM benchmark composed of 1000 short natural video-caption pairs. We also welcome the new models: [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://github.com/rese1f/MovieChat).
- [2024-09] 🎉🎉 We welcome the new tasks [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/) for inference acceleration
- [2024-09] ⚙️️⚙️️️️ We upgrade `lmms-eval` to `0.2.3` with more tasks and features. We support a compact set of language tasks evaluations (code credit to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)), and we remove the registration logic at start (for all models and tasks) to reduce the overhead. Now `lmms-eval` only launches necessary tasks/models. Please check the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3) for more details.
- [2024-08] 🎉🎉 We welcome the new model [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158). We provide new feature of SGlang Runtime API for llava-onevision model, please refer the [doc](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/commands.md) for inference acceleration
- [2024-07] 👨‍💻👨‍💻 The `lmms-eval/v0.2.1` has been upgraded to support more models, including [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA), and many more evaluation tasks, e.g. [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136), [MLVU](https://arxiv.org/abs/2406.04264), [WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data), [VITATECS](https://github.com/lscpku/VITATECS) and [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/).
- [2024-07] 🎉🎉 We have released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)! 
- [2024-06] 🎬🎬 The `lmms-eval/v0.2.0` has been upgraded to support video evaluations for video models like LLaVA-NeXT Video and Gemini 1.5 Pro across tasks such as EgoSchema, PerceptionTest, VideoMME, and more. Please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.2/) for more details!
- [2024-03] 📝📝 We have released the first version of `lmms-eval`, please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.1/) for more details!

</details>

## Why LMMs-Eval?

LMMs-Eval is your all-in-one solution for assessing the capabilities of LMMs, providing a centralized and streamlined platform for evaluating performance on a wide range of multimodal tasks.  We build upon the robust design of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Installation

```bash
# Using uv (Recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

# For development
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

Refer to the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5's paper results. Note that results may vary slightly due to torch/CUDA version differences; see [results check](miscs/llava_result_check.md).

</details>

If you need to test caption datasets, ensure you have `java==1.8.0` installed, for the pycocoeval API to function:

```bash
conda install openjdk=8
```

Verify your java version by running `java -version`.

<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

We aim to provide detailed information for readers to understand the datasets included in lmms-eval and some specific details about these datasets (we remain grateful for any corrections readers may have during our evaluation process).

We provide a Google Sheet for the detailed results of the LLaVA series models on different datasets. You can access the sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing). It's a live sheet, and we are updating it with new results.

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

We also provide the raw data exported from Weights & Biases for the detailed results of the LLaVA series models on different datasets. You can access the raw data [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>
<br>

If you wish to test [VILA](https://github.com/NVlabs/VILA), install its dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

We are actively developing LMMs-Eval on the `main` branch. We encourage your feedback and contributions to improve the library. Please submit issues or pull requests on GitHub.

## Usage Examples

> Explore more examples in the [examples/models](examples/models) directory.

**OpenAI-Compatible Model Evaluation**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**vLLM Evaluation**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**LLaVA-OneVision Evaluation**

```bash
bash examples/models/llava_onevision.sh
```

**Llama-3-Vision Evaluation**

```bash
bash examples/models/llama_vision.sh
```

**Qwen2-VL Evaluation**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**LLaVA on MME Evaluation**

If you want to test LLaVA 1.5, you will have to clone their repo from [LLaVA](https://github.com/haotian-liu/LLaVA) and

```bash
bash examples/models/llava_next.sh
```

**Tensor Parallel Evaluation (for larger models, e.g., llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**SGLang Evaluation (for larger models, e.g., llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**vLLM Evaluation (for larger models, e.g., llava-next-72b)**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Additional Parameters**

```bash
python3 -m lmms_eval --help
```

**Environment Variables**

Set the following environment variables before running evaluations:

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include 
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Common Environment Issues**

Resolve potential `httpx` and `protobuf` errors with the following:

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install numpy==1.26;
# Someties sentencepiece are required for tokenizer to work
python3 -m pip install sentencepiece;
```

## Adding Custom Models and Datasets

Refer to the detailed [documentation](docs/README.md) for guidance.

## Acknowledgments

LMMs-Eval is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).  Consult their [docs](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for valuable insights.

---

Below are the changes we made to the original API:
- Build context now only pass in idx and process image and doc during the model responding phase. This is due to the fact that dataset now contains lots of images and we can't store them in the doc like the original lm-eval-harness other wise the cpu memory would explode.
- Instance.args (lmms_eval/api/instance.py) now contains a list of images to be inputted to lmms.
- lm-eval-harness supports all HF language models as single model class. Currently this is not possible of lmms because the input/output format of lmms in HF are not yet unified. Thererfore, we have to create a new class for each lmms model. This is not ideal and we will try to unify them in the future.
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