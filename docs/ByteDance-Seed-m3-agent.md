<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: A Multimodal Agent with Human-Like Long-Term Memory

**Unlock the future of AI with M3-Agent, a multimodal agent that sees, listens, remembers, and reasons to achieve impressive results.**

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**[View the original repository](https://github.com/ByteDance-Seed/m3-agent)**

## Key Features

*   **Multimodal Understanding:** Processes real-time visual and auditory inputs for a comprehensive understanding of the environment.
*   **Long-Term Memory:** Builds and updates a long-term memory, including both episodic and semantic knowledge, mirroring human memory capabilities.
*   **Entity-Centric Organization:** Organizes memory in a multimodal, entity-centric format for deeper and more consistent understanding.
*   **Iterative Reasoning:** Autonomously performs multi-turn, iterative reasoning to accomplish tasks.
*   **Superior Performance:** Outperforms leading baselines on the M3-Bench benchmark.
*   **M3-Bench Dataset:** Features M3-Bench, a new long-video question answering benchmark, to evaluate memory effectiveness and reasoning in multimodal agents.

## M3-Bench: Evaluating Long-Term Memory in Multimodal Agents

M3-Bench is a key component for evaluating the performance of M3-Agent. It includes:

*   **M3-Bench-robot:** 100 real-world videos from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos across diverse scenarios.
*   Question-answer pairs designed to test human understanding, general knowledge extraction, and cross-modal reasoning.

![architecture](figs/m3-bench-example.png)
*Examples from M3-Bench, showcasing M3-Bench-robot and M3-Bench-web.*

![architecture](figs/m3-bench-statistic.png)
*Statistical overview of M3-Bench benchmark.*

### Download M3-Bench

*   **M3-Bench-robot:** [Hugging Face Datasets](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
*   **M3-Bench-web:**  Find video URLs in `data/annotations/web.json`.

### Intermediate Outputs & Memory Graphs (Optional)

*   Download pre-processed data from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) and [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) or generate them using the steps below.

## M3-Agent Architecture

![architecture](figs/m3-agent.png)

The M3-Agent architecture includes:

*   **Memorization Process:** Generates episodic and semantic memory from video and audio streams.
*   **Control Process:** Executes instructions by reasoning and retrieving information from long-term memory.
*   **Multimodal Graph:** Long-term memory is structured as a multimodal graph.

## Experimental Results

![architecture](figs/exp_result.png)

*Performance of M3-Agent on M3-Bench-robot, M3-Bench-web, and VideoMME-long.*

## Run Locally

### Prerequisites

*   Add API config in `configs/api_config.json`.
*   Set up environment using `setup.sh`.
*   Install required packages.

### Memorization

1.  **Cut Video:** Split videos into 30-second segments.
2.  **Prepare Data:** Create a `data/data.jsonl` file with video information.
3.  **Generate Intermediate Outputs:** This step uses Face Detection and Speaker Diarization tools.  Download audio embedding model ([pretrained_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt)) and speakerlab ([speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab)).
    ```bash
    python m3_agent/memorization_intermediate_outputs.py \
       --data_file data/data.jsonl
    ```
4.  **Generate Memory Graphs:** Download M3-Agent-Memorization from [huggingface](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
    ```bash
    python m3_agent/memorization_memory_graphs.py \
       --data_file data/data.jsonl
    ```
5.  **Memory Graph Visualization:** Visualize the generated memory graphs.

    ```bash
    python visualization.py \
       --mem_path data/memory_graphs/robot/bedroom_01.pkl \
       --clip_id 1
    ```

### Control

1.  **Set Up Environment:** Install the required packages.
2.  **Question Answering and Evaluation:** Download M3-Agent-Control from [huggingface](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/blob/main/videos/robot).

    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### Alternative Models

Adapt model inference with api calling and specific prompts.

*   **Memorization Prompts:** Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`; Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`
*   **Control Prompts:** GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

1.  Memorization: [https://github.com/hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
2.  Control: [https://github.com/hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

```BibTeX
@misc{long2025seeing,
      title={Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory}, 
      author={Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, Wei Li},
      year={2025},
      eprint={2508.09736},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```