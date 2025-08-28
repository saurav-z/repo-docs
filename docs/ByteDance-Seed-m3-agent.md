<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Human-Like Long-Term Memory</h1>

**M3-Agent enables agents to see, hear, remember, and reason, advancing the field of multimodal AI.** [Explore the M3-Agent Repository](https://github.com/ByteDance-Seed/m3-agent)

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features of M3-Agent:

*   **Multimodal Understanding:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates both episodic and semantic memory, mirroring human cognition.
*   **Entity-Centric Memory:** Organizes memory in a multimodal, entity-centric format for a deeper understanding of the environment.
*   **Iterative Reasoning:**  Performs multi-turn reasoning, retrieving relevant information from memory to complete tasks.
*   **Superior Performance:** Outperforms leading baselines on the M3-Bench benchmark.

## M3-Bench:  A Benchmark for Long-Term Memory Reasoning

M3-Bench is a novel benchmark designed to evaluate multimodal agents' reasoning abilities over long-term memory. It features:

*   **Diverse Video Data:**  Includes M3-Bench-robot (robot's perspective) and M3-Bench-web (web-sourced videos).
*   **Comprehensive Question Answering:**  Uses question-answer pairs designed to assess memory effectiveness and reasoning capabilities.
*   **Evaluation of Key Capabilities:** Focuses on understanding, knowledge extraction, and cross-modal reasoning.

**Example Videos:**
[Link1](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [Link2](https://www.youtube.com/watch?v=Efk3K4epEzg), [Link3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)

**M3-Bench Dataset:**
*   **M3-Bench-robot:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
*   **M3-Bench-web:** Access video URLs in `data/annotations/web.json`.

**Intermediate Outputs and Memory Graphs:** Download intermediate outputs and memory graphs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) or generate them.

## M3-Agent Architecture

![architecture](figs/m3-agent.png)

M3-Agent's architecture consists of parallel memorization and control processes, enabling efficient information processing and task execution.

## Experimental Results

![architecture](figs/exp_result.png)

M3-Agent demonstrates state-of-the-art performance on M3-Bench and VideoMME-long benchmarks.

## Getting Started: Run M3-Agent Locally

### Prerequisites

*   Ensure you have the required `api_config.json` file configured in the `configs/` directory.

### 1. Memorization

The following steps are required only if you haven't downloaded *intermediate_outputs* and *memory_graphs* from huggingface or want to process other videos not from M3-Bench.

1.  **Set Up Environment:**
    ```bash
    bash setup.sh
    pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
    pip install qwen-omni-utils==0.0.4
    ```
2.  **Cut Videos:** Prepare your videos in 30-second segments using the provided script.
3.  **Prepare Data:** Create a `data/data.jsonl` file with video information.
4.  **Generate Intermediate Outputs:** Run `memorization_intermediate_outputs.py` (requires additional setup detailed in the original README).
    *   Download audio embedding model from [pretrained_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt) and save into `models\`
    *   Download [speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab)
    ```
    m3-agent
    ├── models
    │   └── pretrained_eres2netv2.ckpt
    └── speakerlab
    ```
5.  **Generate Memory Graphs:** Use the `memorization_memory_graphs.py` script.  Download M3-Agent-Memorization from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
6.  **Memory Graph Visualization:** Visualize generated memory graphs using the `visualization.py` script.

### 2. Control

1.  **Set Up Environment:**
    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```
2.  **Question Answering and Evaluation:** Run the `control.py` script to generate answers. Download M3-Agent-Control from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/blob/main/videos/robot).

### Other Models

Easily integrate other models by adapting the model inference to API calls and using corresponding prompts.

### Prompts:

1.  **Memorization:**
    *   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`
2.  **Control:**
    *   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

1.  **Memorization:** [https://github.com/hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
2.  **Control:** [https://github.com/hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

## Citation

If you use M3-Agent in your research, please cite our paper:

```BibTeX
@misc{long2025seeing,
      title={Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory}, 
      author={Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, Wei Li},
      year={2025},
      eprint={2508.09736},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}