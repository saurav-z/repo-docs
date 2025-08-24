<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

# M3-Agent: The Multimodal Agent with Human-Like Long-Term Memory

**M3-Agent revolutionizes multimodal AI by equipping agents with advanced long-term memory for more human-like understanding and reasoning.**  ([Original Repo](https://github.com/ByteDance-Seed/m3-agent))

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features of M3-Agent:

*   **Multimodal Perception:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:**  Builds and updates episodic and semantic memory.
*   **Entity-Centric Memory:**  Organizes memory in a multimodal graph format for deeper understanding.
*   **Iterative Reasoning:** Performs multi-turn reasoning to accomplish tasks.
*   **M3-Bench Benchmark:** Evaluates memory effectiveness and reasoning capabilities.

## M3-Bench: Evaluating Long-Term Memory in Multimodal Agents

M3-Bench is a comprehensive benchmark designed to assess a multimodal agent's ability to reason over long-term memory using long videos.

*   **M3-Bench-robot:** 100 videos recorded from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos covering diverse scenarios.
*   **Question-Answer Pairs:** Designed to test human understanding, knowledge extraction, and cross-modal reasoning.

Explore M3-Bench: [Examples of M3-Bench](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [link2](https://www.youtube.com/watch?v=Efk3K4epEzg), [link3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)

## Architecture and Experimental Results
![architecture](figs/m3-agent.png)
![architecture](figs/exp_result.png)

## Run M3-Agent Locally

Follow these steps to run M3-Agent and experiment with its capabilities.

### Memorization

1.  **Setup Environment:**
    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```
2.  **Cut Videos (Optional):**
    Cut videos into 30-second segments using the provided `cut_video.sh` script.
3.  **Prepare Data:**
    Create a `data/data.jsonl` file with video information.  See the README for an example.
4.  **Generate Intermediate Outputs (Optional):**  Download and install speakerlab, and audio embedding models or follow the guide provided.
    ```bash
    python m3_agent/memorization_intermediate_outputs.py \
       --data_file data/data.jsonl
    ```
5.  **Generate Memory Graphs:**  Download M3-Agent-Memorization or follow the guide provided.
    ```bash
    python m3_agent/memorization_memory_graphs.py \
       --data_file data/data.jsonl
    ```
6.  **Memory Graph Visualization (Optional):**
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
2.  **Question Answering and Evaluation:** Download M3-Agent-Control.
    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### Other Models

*   Prompts for Memorization, Control and API information are listed in the original README

## Training

*   Memorization: [Link to Training Repo](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   Control: [Link to Training Repo](https://github.com/hyc2026/M3-Agent-Training)

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