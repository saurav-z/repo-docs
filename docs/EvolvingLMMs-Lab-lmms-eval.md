<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: Evaluate and Advance Large Multimodal Models

**LMMs-Eval is a powerful evaluation framework designed to streamline the development and assessment of Large Multimodal Models (LMMs), supporting diverse tasks across text, image, video, and audio.** Explore the original repository on [GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval).

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features

*   **Comprehensive Task Support:** Evaluates LMMs across a wide array of modalities including text, image, video, and audio.
*   **Extensive Model Compatibility:** Supports over 30 different LMMs, with new models constantly being added.
*   **Accelerated Evaluation:** Integrates technologies like vLLM and OpenAI API compatibility for faster model assessment.
*   **Reproducibility Focus:**  Provides resources and scripts to reproduce results, ensuring reliability and transparency.
*   **Modular Design:**  Inspired by the successful `lm-evaluation-harness`, offering a flexible and efficient evaluation framework.
*   **Active Development:**  Continuously updated with new benchmarks, models, and features, driven by community contributions.

---

## Announcements
- [2025-07] 🚀🚀  `lmms-eval-0.4` Released! Includes major updates and improvements. See [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md).
- [2025-04] 🚀🚀 Introducing Aero-1-Audio — compact yet mighty audio model with support for batched evaluations.
- [2025-07] 🎉🎉 New task [PhyX](https://phyx-bench.github.io/), the first large-scale benchmark assessing physics-grounded reasoning in visual scenarios.
- [2025-06] 🎉🎉 New task [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), designed to evaluate mathematical reasoning in real-world educational videos.
- [2025-02] 🚀🚀  Integrated [`vllm`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/544) and [`openai_compatible`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/546) for accelerated and API-based model evaluation.

<details>
<summary>Below is a chronological list of recent tasks, models, and features added by our amazing contributors. </summary>

- [2025-01] 🎓🎓 We have released our new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826). Please refer to the [project page](https://videommmu.github.io/) for more details.
- [2024-12] 🎉🎉 We have presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296), jointly with [MME Team](https://github.com/BradyFU/Video-MME) and [OpenCompass Team](https://github.com/open-compass).
- [2024-11] 🔈🔊 The `lmms-eval/v0.3.0` has been upgraded to support audio evaluations for audio models like Qwen2-Audio and Gemini-Audio across tasks such as AIR-Bench, Clotho-AQA, LibriSpeech, and more. Please refer to the [blog](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.3.md) for more details!
- [2024-10] 🎉🎉 We welcome the new task [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), a vision-centric VQA benchmark (NeurIPS'24) that challenges vision-language models with simple questions about natural imagery.
- [2024-10] 🎉🎉 We welcome the new task [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) for fine-grained temporal understanding and reasoning for videos, which reveals a huge (>30%) human-AI gap.
- [2024-10] 🎉🎉 We welcome the new tasks [VDC](https://rese1f.github.io/aurora-web/) for video detailed captioning, [MovieChat-1K](https://rese1f.github.io/MovieChat/) for long-form video understanding, and [Vinoground](https://vinoground.github.io/), a temporal counterfactual LMM benchmark composed of 1000 short natural video-caption pairs. We also welcome the new models: [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://rese1f/MovieChat).
- [2024-09] 🎉🎉 We welcome the new tasks [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/) for inference acceleration
- [2024-09] ⚙️️⚙️️️️ We upgrade `lmms-eval` to `0.2.3` with more tasks and features. We support a compact set of language tasks evaluations (code credit to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)), and we remove the registration logic at start (for all models and tasks) to reduce the overhead. Now `lmms-eval` only launches necessary tasks/models. Please check the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3) for more details.
- [2024-08] 🎉🎉 We welcome the new model [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158). We provide new feature of SGlang Runtime API for llava-onevision model, please refer the [doc](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/commands.md) for inference acceleration
- [2024-07] 👨‍💻👨‍💻 The `lmms-eval/v0.2.1` has been upgraded to support more models, including [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA), and many more evaluation tasks, e.g. [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136), [MLVU](https://arxiv.org/abs/2406.04264), [WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data), [VITATECS](https://github.com/lscpku/VITATECS) and [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/).
- [2024-07] 🎉🎉 We have released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)! 
- [2024-06] 🎬🎬 The `lmms-eval/v0.2.0` has been upgraded to support video evaluations for video models like LLaVA-NeXT Video and Gemini 1.5 Pro across tasks such as EgoSchema, PerceptionTest, VideoMME, and more. Please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.2/) for more details!
- [2024-03] 📝📝 We have released the first version of `lmms-eval`, please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.1/) for more details!

</details>

---

## Why LMMs-Eval?

LMMs-Eval provides a centralized and efficient solution for evaluating the capabilities of cutting-edge Large Multimodal Models. The framework is inspired by `lm-evaluation-harness`, enabling researchers and developers to quickly and consistently benchmark their models across various tasks and modalities.

---

## Installation

**From Source (Recommended):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

**For Development:**

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

<details>
<summary>Reproducing LLaVA-1.5 Results</summary>

See [miscs/repr_scripts.sh](miscs/repr_scripts.sh) and [miscs/repr_torch_envs.txt](miscs/repr_torch_envs.txt) to reproduce LLaVA-1.5 results. Results check in [miscs/llava_result_check.md].

</details>

**Additional Dependencies:**

*   For `coco`, `refcoco`, and `nocaps` datasets: `conda install openjdk=8` and check version with `java -version`.
*   For VILA: `pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales`

---

## Usage Examples

*   **OpenAI-Compatible Models:**
    ```bash
    bash examples/models/openai_compatible.sh
    bash examples/models/xai_grok.sh
    ```

*   **vLLM Evaluation:**
    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

*   **LLaVA-OneVision:**
    ```bash
    bash examples/models/llava_onevision.sh
    ```

*   **Llama-3-Vision:**
    ```bash
    bash examples/models/llama_vision.sh
    ```

*   **Qwen2-VL:**
    ```bash
    bash examples/models/qwen2_vl.sh
    bash examples/models/qwen2_5_vl.sh
    ```

*   **LLaVA on MME (Requires LLaVA repo):**
    ```bash
    bash examples/models/llava_next.sh
    ```

*   **Tensor Parallel Evaluation (llava-next-72b):**
    ```bash
    bash examples/models/tensor_parallel.sh
    ```

*   **SGLang Evaluation (llava-next-72b):**
    ```bash
    bash examples/models/sglang.sh
    ```

*   **vLLM Evaluation (llava-next-72b):**
    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

**More Parameters:**

```bash
python3 -m lmms_eval --help
```

**Environment Variables:**

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Troubleshooting:**

Address common issues like httpx and protobuf errors:

```bash
python3 -m pip install httpx==0.23.3
python3 -m pip install protobuf==3.20
python3 -m pip install numpy==1.26 # For numpy 2.x issues
python3 -m pip install sentencepiece # For tokenizer issues
```

---

## Contribute

We welcome contributions! Please refer to our [documentation](docs/README.md) for information on adding custom models and datasets. Share your ideas and feedback through issues and pull requests on GitHub.

---

## Acknowledgements

LMMs-Eval is inspired by the design of `lm-evaluation-harness`. Explore its documentation [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs).

---

## Technical Details
The following changes were made to the original API:
-   Context now only passes idx and processes image/doc during the model response phase. This avoids memory issues.
-   Instance.args (lmms_eval/api/instance.py) contains images to be input to the model.
-   New class created for each LMMS model due to format differences of models in HF.

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
```