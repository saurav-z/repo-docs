<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: A Multimodal Agent with Long-Term Memory

**M3-Agent revolutionizes how agents understand and interact with the world by enabling them to see, hear, remember, and reason.** ([Original Repo](https://github.com/ByteDance-Seed/m3-agent))

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features

*   **Multimodal Understanding:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates an entity-centric, multimodal memory.
*   **Semantic Knowledge:** Accumulates world knowledge over time.
*   **Iterative Reasoning:** Performs multi-turn reasoning and information retrieval.
*   **Enhanced Performance:** Outperforms state-of-the-art models on the M3-Bench benchmark.
*   **Open-Source Models & Dataset:** Access to models and M3-Bench for research and application.

## What is M3-Agent?

M3-Agent is a cutting-edge multimodal agent framework designed to mimic human-like understanding and reasoning. This innovative system processes visual and auditory inputs to build and update its long-term memory, leading to a deeper and more consistent comprehension of its environment.

### M3-Bench: Evaluating Long-Term Memory and Reasoning

To assess the effectiveness of memory and reasoning in multimodal agents, we've developed M3-Bench, a new benchmark for long video question answering.

*   **M3-Bench-robot:** 100 real-world videos captured from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos covering diverse scenarios.

**[Example Videos](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [Video 2](https://www.youtube.com/watch?v=Efk3K4epEzg), [Video 3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)**

![architecture](figs/m3-bench-example.png)

Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types.

### Accessing the Data

*   **M3-Bench-robot:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
*   **M3-Bench-web:** Access video URLs in `data/annotations/web.json`.
*   **Intermediate Outputs:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) or generate them.
*   **Memory Graphs:** Download and extract from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) or generate them.

## M3-Agent Architecture

![architecture](figs/m3-agent.png)

M3-Agent's architecture comprises two parallel processes: memorization and control. During memorization, it processes video and audio streams, creating episodic and semantic memory. In control mode, it executes instructions by iteratively thinking and retrieving from long-term memory. The long-term memory is structured as a multimodal graph.

## Experimental Results

![architecture](figs/exp_result.png)

M3-Agent achieves superior performance on M3-Bench-robot, M3-Bench-web, and VideoMME-long.

## Running M3-Agent Locally

### Prerequisites:

1.  **API Configuration:** Add your API configurations to `configs/api_config.json`.
2.  **Setup Environment:**

    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

### Memorization

Generate memory graphs for each video (results saved in `data/memory_graphs`).

#### Steps for Generating Intermediate Outputs & Memory Graphs:

1.  **Video Segmentation:** Cut videos into 30-second segments using `ffmpeg`.
2.  **Prepare Data:** Create a JSONL file (`data/data.jsonl`) listing video details.
3.  **Generate Intermediate Outputs:** (Only if not downloading from Hugging Face)

    *   Download the audio embedding model and save into `models\` from [pretrained_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt)
    *   Download [speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab)

    ```bash
    python m3_agent/memorization_intermediate_outputs.py \
        --data_file data/data.jsonl
    ```

4.  **Generate Memory Graphs:**

    *   Download M3-Agent-Memorization from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)

    ```bash
    python m3_agent/memorization_memory_graphs.py \
        --data_file data/data.jsonl
    ```

5.  **Visualize Memory Graphs:**

    ```bash
    python visualization.py \
        --mem_path data/memory_graphs/robot/bedroom_01.pkl \
        --clip_id 1
    ```

### Control

1.  **Setup Environment:**

    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  **Question Answering and Evaluation:**

    *   Download M3-Agent-Control from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)

    ```bash
    python m3_agent/control.py \
        --data_file data/annotations/robot.json
    ```

### Using Other Models

Adapt API calls and prompts for different models.

*   **Memorization Prompts:**
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`
*   **Control Prompts:**
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

*   Memorization: [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   Control: [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

If you use M3-Agent in your research, please cite us:

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