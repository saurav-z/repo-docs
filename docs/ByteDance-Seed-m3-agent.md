<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: The Multimodal Agent with Long-Term Memory for Enhanced Reasoning

M3-Agent revolutionizes how agents understand and interact with the world by combining sight, sound, memory, and reasoning, achieving state-of-the-art performance in multimodal tasks. For more details, check out the original repository: [ByteDance-Seed/m3-agent](https://github.com/ByteDance-Seed/m3-agent).

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features

*   **Multimodal Processing:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates an entity-centric multimodal memory, including episodic and semantic knowledge.
*   **Advanced Reasoning:** Employs iterative reasoning and retrieval from memory to perform tasks.
*   **Enhanced Performance:** Achieves superior results compared to baseline models on the M3-Bench benchmark.
*   **Open Source:** Source code and pretrained models are publicly available for research and development.

## Overview

M3-Agent is a novel multimodal agent framework designed to mimic human-like cognitive abilities. It incorporates long-term memory to process visual and auditory inputs, enabling it to build and update its understanding of the world continuously. The agent's architecture includes processes for memorization and control, allowing it to effectively reason and execute instructions.

## M3-Bench

M3-Bench is a new benchmark designed to assess the capabilities of multimodal agents. It consists of two subsets: M3-Bench-robot (real-world robotic scenarios) and M3-Bench-web (diverse web-sourced videos). The benchmark includes long videos with question-answer pairs designed to evaluate the agent's ability to construct and reason over long-term memory.

### Dataset Details
*   **M3-Bench-robot**: 100 videos recorded from a robot's perspective.
*   **M3-Bench-web**: 920 videos from various web sources.
*   **Question Types:** Questions assess key capabilities, including human understanding, knowledge extraction, and cross-modal reasoning.

### Dataset Resources

*   **M3-Bench-robot Videos:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
*   **M3-Bench-web Videos:** Download from video_url in `data/annotations/web.json`.
*   **Intermediate Outputs:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) or generate them directly.
*   **Memory Graphs:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) or generate them directly.

## Running M3-Agent Locally

Before running M3-Agent, ensure that you configure your API in `configs/api_config.json`.

### Memorization
The following steps are needed if you want to generate intermediate outputs or memory graphs for videos that are not part of the M3-Bench dataset.

1.  **Set up the Environment:**
    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```

2.  **Cut Videos:** Split videos into 30-second segments using the provided script.

3.  **Prepare Data:** Create a `data/data.jsonl` file containing video information.

4.  **Generate Intermediate Outputs:** Use face detection and speaker diarization tools.
    *   Download audio embedding model and save into `models\` from [pretrained_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt)
    *   Download [speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab)
```bash
python m3_agent/memorization_intermediate_outputs.py \
   --data_file data/data.jsonl
```

5.  **Generate Memory Graphs:** Utilize the M3-Agent-Memorization model.
```bash
python m3_agent/memorization_memory_graphs.py \
   --data_file data/data.jsonl
```

6.  **Visualize Memory Graphs:**
```bash
python visualization.py \
   --mem_path data/memory_graphs/robot/bedroom_01.pkl \
   --clip_id 1
```

### Control

1.  **Set up the Environment:**
    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  **Question Answering and Evaluation:**
    *   Download M3-Agent-Control from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
    *   Run question answering and evaluation using the provided script.
```bash
python m3_agent/control.py \
   --data_file data/annotations/robot.json
```

### Prompts

*   **Memorization:** Utilize the provided prompts for Gemini/GPT-4o and Qwen2.5-Omni-7B.
*   **Control:** Use the prompts for GPT-4o to generate answers.

## Training

*   Memorization: Resources available at [sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker).
*   Control: Resources available at [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training).

## Citation

If you use this work, please cite the following:

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