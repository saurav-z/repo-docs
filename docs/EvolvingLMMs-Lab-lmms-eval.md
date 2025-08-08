<p align="center" width="70%">
<img src="https://i.postimg.com/KvkLzbF9/WX20241212-014400-2x.png" alt="LMMs-Eval Logo">
</p>

# LMMs-Eval: Your Comprehensive Toolkit for Evaluating Large Multimodal Models

**Accelerate your LMM development with LMMs-Eval, a versatile framework for assessing and comparing state-of-the-art multimodal AI models.**

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> LMMs-Eval provides a robust, efficient, and continuously updated platform for evaluating Large Multimodal Models (LMMs) across various tasks, including text, image, video, and audio.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Key Features:

*   **Comprehensive Task Support:** Evaluate LMMs on a vast array of tasks spanning text, image, video, and audio modalities.
*   **Extensive Model Compatibility:**  Test your models against a wide selection of supported LMMs.
*   **Efficient Evaluation:** Optimized for speed and accuracy, enabling rapid assessment of model performance.
*   **OpenAI API Compatibility:** Support evaluation of any API-based model that follows the OpenAI API format.
*   **Regular Updates:** Benefit from continuous updates with new tasks, models, and features.
*   **Reproducibility:**  Includes scripts and resources to reproduce results from key research papers.
*   **Community Driven:**  Contribute and collaborate with our active community to enhance the framework.

## What's New

*   **[2024-07]** 🚀🚀 We have released the `lmms-eval-0.4`. Please refer to the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) for more details. This is a major update with new features and improvements, for users wish to use `lmms-eval-0.3` please refer to the branch `stable/v0d3`. For our mission to better reproductability, we've opened a specific thread to discuss about the model's eval results in [discussion](https://github.com/EvolvingLMMs-Lab/lmms-eval/discussions/779).
*   **[2025-04]** 🚀🚀 Introducing Aero-1-Audio — a compact yet mighty audio model. We have officially supports evaluation for Aero-1-Audio and it supports batched evaluations! Feel free to try out.
*   **[2025-07]** 🎉🎉 We welcome the new task [PhyX](https://phyx-bench.github.io/), the first large-scale benchmark designed to assess models capacity for physics-grounded reasoning in visual scenarios.
*   **[2025-06]** 🎉🎉 We welcome the new task [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), designed to evaluate mathematical reasoning in real-world educational videos.
*   **[2025-04]** 🚀🚀 Introducing [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/) — a compact yet mighty audio model. We have officially supports evaluation for Aero-1-Audio and it supports batched evaluations! Feel free to try out.
*   **[2025-02]** 🚀🚀 We have integrated [`vllm`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/544) into our models, enabling accelerated evaluation for both multimodal and language models. Additionally, we have incorporated [`openai_compatible`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/546) to support the evaluation of any API-based model that follows the OpenAI API format. Check the usages [here](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/miscs/model_dryruns).
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

## Installation

Get started quickly with LMMs-Eval:

**Installation using `uv`:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

**Development Installation:**

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

### Additional Notes

*   **Reproducing LLaVA-1.5 Results:** Check the `miscs/repr_scripts.sh` and `miscs/repr_torch_envs.txt` for environment setup.  Review the `miscs/llava_result_check.md` file for results verification across different environments.
*   **COCO, RefCOCO, and NoCaps Dependencies:**  Install `java==1.8.0` using `conda install openjdk=8` to use the pycocoeval API.  Verify the installation with `java -version`.
*   **VILA Dependencies:**  Install `pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales` to test [VILA](https://github.com/NVlabs/VILA).

## Evaluation Results

[Comprehensive Evaluation Results of LLaVA Family Models](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing)
  
<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

[Raw Data](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

## Usage Examples

Explore how to evaluate various models with these example scripts:

*   **OpenAI-Compatible Models:**
    ```bash
    bash examples/models/openai_compatible.sh
    bash examples/models/xai_grok.sh
    ```

*   **vLLM:**
    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

*   **LLaVA-OneVision:**
    ```bash
    bash examples/models/llava_onevision.sh
    ```

*   **LLaMA-3.2-Vision:**
    ```bash
    bash examples/models/llama_vision.sh
    ```

*   **Qwen2-VL:**
    ```bash
    bash examples/models/qwen2_vl.sh
    bash examples/models/qwen2_5_vl.sh
    ```

*   **LLaVA on MME (Clone LLaVA Repo):**
    ```bash
    bash examples/models/llava_next.sh
    ```

*   **Tensor Parallel (llava-next-72b):**
    ```bash
    bash examples/models/tensor_parallel.sh
    ```

*   **SGLang (llava-next-72b):**
    ```bash
    bash examples/models/sglang.sh
    ```

*   **vLLM (llava-next-72b):**
    ```bash
    bash examples/models/vllm_qwen2vl.sh
    ```

### Additional Resources

*   **Command Line Help:** Get a list of available parameters with: `python3 -m lmms_eval --help`
*   **Environment Variables:**  Set environment variables for proper functionality (e.g., `OPENAI_API_KEY`, `HF_HOME`, `HF_TOKEN`, `HF_HUB_ENABLE_HF_TRANSFER`, `REKA_API_KEY`). See the original README for full details.
*   **Troubleshooting:**  Resolve potential installation issues by running:
    ```bash
    python3 -m pip install httpx==0.23.3;
    python3 -m pip install protobuf==3.20;
    python3 -m pip install numpy==1.26;
    python3 -m pip install sentencepiece;
    ```

## Add Custom Models and Datasets

For custom model and dataset integration, consult the [documentation](docs/README.md).

## Acknowledgements

LMMs-Eval is built upon the foundation of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). We recommend reviewing the [lm-eval-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for related information.

---

**Key Modifications to Original API:**

*   Context building now processes images and documents during the model response phase.
*   `Instance.args` now contains a list of images for LMM input.
*   New model classes created for individual LMM models due to differences in HF input/output formats.

## Citations

Cite the following works when referencing LMMs-Eval:

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

**[Contribute to LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)**