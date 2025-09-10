<div align=left>
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width=40%>
</div>

<h1 style="text-align: center;">M3-Agent: A Multimodal Agent with Human-Like Long-Term Memory</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.09736-b31b1b.svg)](https://arxiv.org/abs/2508.09736)
[![Demo](https://img.shields.io/badge/homepage-M3--Agent-blue)](https://m3-agent.github.io)
[![Model](https://img.shields.io/badge/model_HF-Memorization-green)](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)
[![Model](https://img.shields.io/badge/model_HF-Control-darkgreen)](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)
[![Data](https://img.shields.io/badge/data-M3--Bench-F9D371)](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)

M3-Agent is a cutting-edge multimodal agent that sees, listens, remembers, and reasons, advancing the field of AI towards more human-like long-term memory. [Explore the original repo](https://github.com/ByteDance-Seed/m3-agent) for more details.

**Key Features:**

*   **Multimodal Perception:** Processes real-time visual and auditory inputs.
*   **Long-Term Memory:** Builds and updates an entity-centric, multimodal memory.
*   **Semantic Understanding:** Develops world knowledge through semantic memory.
*   **Iterative Reasoning:** Performs multi-turn reasoning and retrieves relevant information.
*   **M3-Bench Benchmark:** Evaluated on a novel long-video question-answering benchmark.
*   **Superior Performance:** Outperforms baseline models on M3-Bench and VideoMME-long.

**What is M3-Agent?**

M3-Agent is a novel multimodal agent framework designed to mimic human-like cognitive abilities. It processes video and audio streams to build and update a long-term memory, allowing it to understand and reason about its environment. The agent's memory is structured in an entity-centric, multimodal format, which allows for deeper and more consistent understanding. M3-Agent can autonomously perform multi-turn, iterative reasoning and retrieve relevant information from memory to accomplish tasks.

**M3-Bench: Evaluating Long-Term Memory**

M3-Bench is a specifically designed long-video question-answering dataset to evaluate the effectiveness of long-term memory and memory-based reasoning in multimodal agents. It consists of two subsets:

*   **M3-Bench-robot:** 100 real-world videos from a robot's perspective.
*   **M3-Bench-web:** 920 web-sourced videos covering diverse scenarios.

**Get Started:**

*   **M3-Bench Dataset:**
    *   Download M3-Bench-robot from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/videos/robot)
    *   Download M3-Bench-web from video\_url in `data/annotations/web.json`

*   **Intermediate Outputs and Memory Graphs:**
    *   Optional: Download pre-processed outputs and graphs from [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/intermediate_outputs) and [Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/tree/main/memory_graphs) respectively. Or generate them.

**Run Locally**

Follow the below steps to get the project up and running.
> Before running, add api config in `configs/api_config.json`

### Memorization

Generate memory graphs for each video. The results are saved in `data/memory_graphs`.

- The following steps are required only if you haven't downloaded *intermediate_outputs* and *memory_graphs* from huggingface or want to process other videos not from M3-Bench.

1.  **Set up environment**

```bash
bash setup.sh
pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
pip install qwen-omni-utils==0.0.4
```

2.  **Cut Video**

    Cut the video into 30-second segments.

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

3.  **Prepare data**

    Prepare a jsonl file with one video per line saved in `data/data.jsonl`

```json
{"id": "bedroom_01", "video_path": "data/videos/robot/bedroom_01.mp4", "clip_path": "data/videos/clips/bedroom_01", "mem_path": "data/videos/memory_graphs/bedroom_01.pkl", "intermediate_path": "data/videos/intermediate_outputs/robot/bedroom_01"}
```

4.  **Generate Intermediate Outputs**

    **This step uses Face Detection and Speaker Diarization tools to generate intermediate outputs.**

    -   If you want to use M3-Bench and have downloaded intermediate\_outputs from huggingface, you can skip this step.

    -   Download audio embedding model and save into `models\` from [pretrained_eres2netv2.ckpt](https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt)

    -   Download [speakerlab](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab)

    ```
    m3-agent
    ├── models
    │   └── pretrained_eres2netv2.ckpt
    └── speakerlab
    ```

```bash
python m3_agent/memorization_intermediate_outputs.py \
   --data_file data/data.jsonl
```

5.  **Generate Memory Graphs**

    **This step uses the M3-Agent-Memorization model to generate memory graphs.**

    -   Download M3-Agent-Memorization from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)

```bash
python m3_agent/memorization_memory_graphs.py \
   --data_file data/data.jsonl
```

6.  **Memory Graph Visualization**

```bash
python visualization.py \
   --mem_path data/memory_graphs/robot/bedroom_01.pkl \
   --clip_id 1
```

### Control

1.  **Set up environment**

```bash
bash setup.sh
pip install transformers==4.51.0
pip install vllm==0.8.4
pip install numpy==1.26.4
```

2.  **Question Answering and Evaluation**

    **This step uses the M3-Agent-Control model to generate answer and the GPT-4o to evaluate the answer.**

    -   Download M3-Agent-Control from [huggingface](https://huggingface.co/ByteDance-Seed/M3-Agent-Control)

```bash
python m3_agent/control.py \
   --data_file data/annotations/robot.json
```

### Other Models

If you want to prompt other models to generate memory or answer question, only need to change the model inference into api calling and use the corresponding prompt.

Prompts:

1.  Memorization

    -   Gemini/GPT-4o: `mmagent.prompts.prompt_generate_captions_with_ids`
    -   Qwen2.5-Omni-7B: `mmagent.prompts.prompt_generate_full_memory`
2.  Control

    -   GPT-4o: `mmagent.prompts.prompt_answer_with_retrieval_final`

**Training**

*   Memorization: [https://github.com/hyc2026/sft-qwen2.5-omni-thinker](https://github.com/hyc2026/sft-qwen2.5-omni-thinker)
*   Control: [https://github.com/hyc2026/M3-Agent-Training](https://github.com/hyc2026/M3-Agent-Training)

**Citation**

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