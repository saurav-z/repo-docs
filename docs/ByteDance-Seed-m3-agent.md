<div align="left">
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Long-Term Memory for Enhanced Reasoning</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

**M3-Agent revolutionizes multimodal understanding by equipping AI with long-term memory, enabling human-like reasoning and knowledge retention.**  For the original source code, please visit the [M3-Agent GitHub Repository](https://github.com/ByteDance-Seed/m3-agent).

## Key Features

*   **Multimodal Perception:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates both episodic and semantic memory.
*   **Entity-Centric Organization:** Stores memory in a structured, multimodal graph for deeper understanding.
*   **Iterative Reasoning:** Performs multi-turn reasoning, retrieving relevant information from memory to complete tasks.
*   **M3-Bench Dataset:**  Includes M3-Bench-robot and M3-Bench-web for evaluating long-term memory and reasoning capabilities.
*   **Superior Performance:** Outperforms baseline models on the M3-Bench dataset.

## What is M3-Agent?

M3-Agent is a cutting-edge multimodal agent designed to mimic human-like cognitive abilities. It goes beyond simply processing information; it learns, remembers, and reasons.  The agent's architecture consists of two main components:  memorization and control. The memorization process transforms incoming video and audio data into both episodic and semantic memories. During control, the agent tackles instructions by iteratively thinking and retrieving from its long-term memory, represented as a multimodal graph.

## M3-Bench Dataset

M3-Agent's performance is validated using the M3-Bench dataset, designed to test a multimodal agent's capacity to perform reasoning based on long-term memory.

*   **M3-Bench-robot:** Features 100 real-world videos captured from a robot's perspective.
*   **M3-Bench-web:** Contains 920 web-sourced videos covering a diverse range of scenarios.
*   **Comprehensive Evaluation:** Includes question-answer pairs designed to assess key agent capabilities such as human understanding, general knowledge extraction, and cross-modal reasoning.

**(See example images from M3-Bench in the original README)**

## Run Locally

This section provides instructions for setting up and running the M3-Agent locally, including instructions for generating intermediate outputs, memory graphs and controlling the model.  Before you begin, ensure that you have added your API configuration in `configs/api_config.json`.

### Memorization

*   **Prerequisites**: Before running, download the necessary models and setup the environment, as detailed in the "Run Locally" section of the original README. You can skip steps if you have already downloaded intermediate outputs and memory graphs from Hugging Face.

*   **Steps**:
    1.  Set up the environment using `bash setup.sh` and install necessary Python packages (see original README).
    2.  Cut Videos using the provided `ffmpeg` script.
    3.  Prepare a data.jsonl file as instructed in the original README.
    4.  Generate Intermediate Outputs, using the face detection and speaker diarization tools, or download from Hugging Face.
    5.  Generate Memory Graphs using the M3-Agent-Memorization model, or download from Hugging Face.
    6.  Visualize the generated memory graphs.

### Control

1.  Set up the environment as instructed in the "Run Locally" section of the original README.
2.  Perform question answering and evaluation, using the M3-Agent-Control model.

### Other Models

The framework is flexible enough to allow the use of other models.  Simply change the model inference to API calls, and use the corresponding prompts.

## Training

The original README provides links to the training code for Memorization and Control models, in the links provided.

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