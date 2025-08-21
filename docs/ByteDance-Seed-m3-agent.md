<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: Building Human-Like Long-Term Memory in Multimodal Agents</h1>

**M3-Agent** revolutionizes multimodal AI, enabling agents to see, hear, remember, and reason like humans.  Explore the technology on the [original GitHub repository](https://github.com/ByteDance-Seed/m3-agent).

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

## Key Features of M3-Agent:

*   **Multimodal Understanding:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates both episodic and semantic memory.
*   **Entity-Centric Memory:** Organizes memory in a multimodal graph for deeper understanding.
*   **Iterative Reasoning:**  Performs multi-turn reasoning to complete tasks.
*   **Superior Performance:** Achieves higher accuracy on benchmarks compared to leading baselines.

## M3-Bench: A Benchmark for Multimodal Reasoning

M3-Bench is a comprehensive dataset designed to evaluate the effectiveness of multimodal agents in reasoning over long-term memory.  It includes two subsets:

*   **M3-Bench-robot:** 100 real-world videos from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos covering diverse scenarios.

**[Illustrative Examples from M3-Bench:**](https://www.youtube.com/watch?v=7W0gRqCRMZQ), [link2](https://www.youtube.com/watch?v=Efk3K4epEzg), [link3](https://www.youtube.com/watch?v=6Unxpxy-Ct4)

![architecture](figs/m3-bench-example.png)
Statistical overview of M3-Bench benchmark. Each question may correspond to multiple question types.

### Accessing M3-Bench Data:

*   **M3-Bench-robot Videos:** Download from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot).
*   **M3-Bench-web Videos:** Access via `video_url` in `data/annotations/web.json`.
*   **Intermediate Outputs & Memory Graphs:**  Download pre-processed data from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) or generate them from the videos (instructions below).

## M3-Agent Architecture

![architecture](figs/m3-agent.png)

M3-Agent utilizes a dual-process architecture:

*   **Memorization:** Processes video and audio to create episodic and semantic memory.
*   **Control:** Executes instructions through iterative reasoning and retrieval from long-term memory.  Long-term memory is structured as a multimodal graph.

## Experimental Results

![architecture](figs/exp_result.png)

M3-Agent demonstrates state-of-the-art performance on M3-Bench and other benchmarks.

## Run M3-Agent Locally

Follow these steps to set up and run M3-Agent:

*   **Important:** Add your API configuration to `configs/api_config.json` before running.

### 1.  Memorization

Generate memory graphs: Results saved in `data/memory_graphs`.

*   **Prerequisites:**

    1.  Set up the environment:
        ```bash
        bash setup.sh
        pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
        pip install qwen-omni-utils==0.0.4
        ```
    2.  Cut Videos: (Use `ffmpeg` to segment videos into 30-second clips.)
        ```bash
        #!/bin/bash
        video="robot/bedroom_01"
        input="data/videos/$video.mp4"
        mkdir -p "data/clips/$video"
        duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input")
        duration_seconds=$(echo "$duration" | awk '{print int($1)}')
         
        segments=$((duration_seconds / 30 + 1))
        for ((i=0; i<segments; i++)); do
            start=$((i * 30))
            end=$(((i + 1) * 30))
            output="data/clips/$video/$i.mp4"
            ffmpeg -ss $start -i "$input" -t 30 -c copy "${output}"
        done
        ```

    3.  Prepare Data: Create a `data/data.jsonl` file (one video per line) in JSONL format:
        ```json
        {"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
        ```

    4.  Generate Intermediate Outputs (Face Detection & Speaker Diarization):

        *   **Important:**  Skip this step if you've downloaded `intermediate_outputs` from Hugging Face.
        *   Download audio embedding models from [pretrained_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt) and save into `models\`.
        *   Download and set up [speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab).

        ```bash
        python m3_agent/memorization_intermediate_outputs.py \
           --data_file data/data.jsonl
        ```

    5.  Generate Memory Graphs:  (Uses M3-Agent-Memorization model)

        *   **Important:** Download M3-Agent-Memorization from [huggingface](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)

        ```bash
        python m3_agent/memorization_memory_graphs.py \
           --data_file data/data.jsonl
        ```

    6.  Memory Graph Visualization:
        ```bash
        python visualization.py \
           --mem_path data/memory_graphs/robot/bedroom_01.pkl \
           --clip_id 1
        ```

### 2. Control

1.  Set up the environment:

    ```bash
    bash setup.sh
    pip install transformers==4.51.0
    pip install vllm==0.8.4
    pip install numpy==1.26.4
    ```

2.  Question Answering and Evaluation:

    *   **Important:**  Download M3-Agent-Control from [huggingface](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/blob/main/videos/robot)

    ```bash
    python m3_agent/control.py \
       --data_file data/annotations/robot.json
    ```

### Alternative Models

You can adapt the code to use other models for memory generation or question answering by modifying the API calls and prompts.

*   **Prompts:**
    *   **Memorization:**
        *   Gemini/GPT-4o:  `mmagent.prompts.prompt_generate_captions_with_ids`
        *   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`
    *   **Control:**
        *   GPT-4o:  `mmagent.prompts.prompt_answer_with_retrieval_final`

## Training

1.  Memorization: Refer to the documentation at [https://github.com/hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker).
2.  Control: See the instructions at [https://github.com/hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training).

## Citation

If you use M3-Agent in your research, please cite our work:

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