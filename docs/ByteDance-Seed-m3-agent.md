<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: Build Long-Term Memory for Multimodal Reasoning</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2508.09736"><img src="https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg" alt="arXiv"></a>
  <a href="https://m3-agent.github.io"><img src="https://img.shields.io/badge/homepage-M3--Agent-blue" alt="Homepage"></a>
  <a href="https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization"><img src="https://img.shields.io/badge/model_HF-Memorization-green" alt="Memorization Model"></a>
  <a href="https://huggingface.co/ByteDance-Seed/M3-Agent-Control"><img src="https://img.shields.io/badge/model_HF-Control-darkgreen" alt="Control Model"></a>
  <a href="https://huggingface.co/datasets/ByteDance-Seed/M3-Bench"><img src="https://img.shields.io/badge/data-M3--Bench-F9D371" alt="M3-Bench Dataset"></a>
</p>

**M3-Agent is a cutting-edge multimodal agent that learns and reasons from visual and auditory inputs, building a long-term memory for enhanced understanding and task execution.** ([Original Repo](https://github.com/ByteDance-Seed/m3-agent))

**Key Features:**

*   **Multimodal Long-Term Memory:** Processes real-time video and audio to construct and update both episodic and semantic memory.
*   **Entity-Centric Memory Organization:**  Stores information in a multimodal graph format, enabling deeper and more consistent environmental understanding.
*   **Iterative Reasoning:**  Performs multi-turn reasoning and retrieves relevant information from memory to accomplish complex tasks.
*   **M3-Bench Dataset:**  A novel benchmark designed to evaluate multimodal agents' ability to reason over long-term memory, featuring both real-world (robot-captured) and web-sourced videos.
*   **State-of-the-Art Performance:** Outperforms strong baselines, demonstrating significant accuracy gains on the M3-Bench and VideoMME-long benchmarks.

**Explore the M3-Agent in Action:**

*   **Demo Video:**  [Watch the video](https://www.youtube.com/watch?v=XUx31cBanfo) showcasing M3-Agent as a personal assistant.  Also available on [Bilibili](https://www.bilibili.com/video/BV1h9YpznEx9/).

**M3-Bench: Evaluating Long-Term Memory and Reasoning**

M3-Bench is a specifically designed dataset for assessing the effectiveness of multimodal agents in tasks requiring long-term memory and reasoning capabilities. The dataset includes:

*   **M3-Bench-robot:**  100 videos captured from a robot's perspective in realistic scenarios.
*   **M3-Bench-web:** 920 web-sourced videos across various content and scenarios.

Each video is accompanied by open-ended question-answer pairs designed to test key abilities like human understanding, general knowledge extraction, and cross-modal reasoning.

<p align="center">
  <img src="figs/m3-bench-example.png" alt="M3-Bench Example" width="70%">
</p>

*   [Example Video 1](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [Example Video 2](https://www.youtube.com/watch?v=Efk3K4epEzg), [Example Video 3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)

<p align="center">
  <img src="figs/m3-bench-statistic.png" alt="M3-Bench Statistics" width="70%">
</p>

**Dataset Resources:**

*   **M3-Bench-robot Videos:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
*   **M3-Bench-web Videos:**  Download video URLs from `data/annotations/web.json`.
*   **Intermediate Outputs:** [Download from Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) (optional), or generate from video clips.
*   **Memory Graphs:**  [Download from Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) (optional), or generate from intermediate outputs.

**M3-Agent Architecture**

<p align="center">
  <img src="figs/m3-agent.png" alt="M3-Agent Architecture" width="70%">
</p>

The M3-Agent architecture comprises two parallel processes: memorization and control.  Memorization builds episodic and semantic memory from visual and audio streams. Control then uses this memory to iteratively reason and execute instructions.

**Experimental Results**

<p align="center">
  <img src="figs/exp_result.png" alt="Experimental Results" width="70%">
</p>

**Run M3-Agent Locally**

1.  **Prerequisites:** Add API configurations in `configs/api_config.json`.
2.  **Memorization:**
    *   Generate memory graphs for each video. Results are saved in `data/memory_graphs`.
    *   **Steps to run memorization:**
        1.  Set up the environment (using `setup.sh`). Install necessary packages, including `transformers` and `vllm` and download pre-trained models.
        2.  Cut videos into 30-second segments using the provided bash script.
        3.  Prepare a `data.jsonl` file with video information.
        4.  Generate intermediate outputs using `m3_agent/memorization_intermediate_outputs.py`.
        5.  Generate memory graphs using `m3_agent/memorization_memory_graphs.py`.
        6.  Visualize memory graphs using `visualization.py`.
3.  **Control:**
    *   Set up the environment. Install necessary packages (same as memorization).
    *   Perform question answering and evaluation using `m3_agent/control.py`.

**Other Models**

Prompt alternative models to generate memory or answer questions by adapting the model inference to API calls and utilizing specific prompts.

*   Memorization Prompts: Gemini/GPT-4o (`mmagent.prompts.prompt_generate_captions_with_ids`), Qwen2.5-Omni-7B (`mmagent.prompts.prompt_generate_full_memory`)
*   Control Prompts: GPT-4o (`mmagent.prompts.prompt_answer_with_retrieval_final`)

**Training**

*   Memorization:  See [SFT-Qwen2.5-Omni-Thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   Control:  See [M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

**Citation**

```bibtex
@misc{long2025seeing,
      title={Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory}, 
      author={Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, Wei Li},
      year={2025},
      eprint={2508.09736},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```