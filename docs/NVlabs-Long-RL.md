<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Long-RL Logo" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Revolutionizing Long Video Reasoning with Reinforcement Learning

**Tackle the challenge of reasoning over extended video sequences with Long-RL, a cutting-edge framework leveraging reinforcement learning to scale vision-language models (VLMs).**  Explore the original repository on [GitHub](https://github.com/NVlabs/Long-RL) for detailed information and resources.

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

## Key Features

*   **Long Video Reasoning:** Enables VLMs to understand and reason about long video sequences.
*   **LongVILA-R1-7B Model:** Achieves state-of-the-art performance on video benchmarks, with up to 8,192 video frames supported, and configurable FPS settings.
*   **Multi-modal RL Training:** Supports RL training on video, text, and audio inputs, accommodating diverse models (VILA, Qwen series, image/video generation models).
*   **Multi-modal Reinforcement Sequence Parallelism (MR-SP):** Provides efficient training infrastructure for long video RL, including sequence parallelism and a vLLM-based engine.
*   **Comprehensive Dataset:** Leverages the LongVideo-Reason dataset, a large-scale dataset with 104K long video QA pairs.
*   **Open-ended Reward Support**: Supports training for open-ended questions.
*   **Cached Video Embeddings**:  Provides the option of using cached video embeddings.
*   **Chunked Gathering**: Offers chunked gathering support to handle potentially large data.

## Highlights

*   **Hour-Level Long Video RL Training:** Train on hour-long videos (3,600 frames - 256k tokens) on a single A100 node (8 GPUs).
*   **Omni-Model RL Support:** Trains RL models that accept text, video, and audio inputs.
*   **Image/Video Generation RL:** Includes support for RL training on image/video generation models like Stable Diffusion and Wan series.

## Installation and Usage

### Installation

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```
If you want to train Qwen-Omni models, please run the command below.
```bash
bash vllm_replace.sh
```

### General Inference

```python
from transformers import AutoModel

model_path = "Efficient-Large-Model/LongVILA-R1-7B"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

# You can adjust the FPS value as needed. 
# To disable FPS control, set it to 0 and manually specify the number of processed video frames via `num_video_frames`.
# Example:
# model.config.fps = 8.0
# model.config.num_video_frames, model.config.fps = 512, 0


use_thinking = True # Switching between thinking and non-thinking modes
system_prompt_thinking = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"

prompt = "What is the main purpose of the video?"
video_path = "video.mp4"

if use_thinking:
  prompt = system_prompt_thinking.format(question=prompt)

response = model.generate_content([prompt, {"path": video_path}])
print("Response: ", response)
```

### Inference with vLLM Engine

Tested on `vllm==0.9.1`.

```bash
mkdir remote_code
cp path_to/Efficient-Large-Model/LongVILA-R1-7B/*.py remote_code
```

```python
import os
from transformers import AutoModel
from vllm import LLM, SamplingParams
from remote_code.media import extract_media
from remote_code.mm_utils import process_images
from remote_code.tokenizer_utils import tokenize_conversation

model_path = "path_to/Efficient-Large-Model/LongVILA-R1-7B"

model_encoder = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto", llm_only_need_embed=True)
# you can change gpu_memory_utilization according to GPU memory
llm = LLM(model=os.path.join(model_path, "llm"), enable_prompt_embeds=True, gpu_memory_utilization=0.5)

use_thinking = True # Switching between thinking and non-thinking modes
system_prompt_thinking = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"

prompt = "What is the main purpose of the video?"
video_path = "video.mp4"

if use_thinking:
  prompt = system_prompt_thinking.format(question=prompt)

conversation = [{"from": "human", "value": [prompt, {"path": video_path}]}]
media = extract_media(conversation, model_encoder.config)
input_ids = tokenize_conversation(conversation, model_encoder.tokenizer, add_generation_prompt=True).unsqueeze(0).cuda()
media["video"] = [
    process_images(images, model_encoder.vision_tower.image_processor, model_encoder.config).half()
    for images in media["video"]
]

inputs_embeds, _, _ = model_encoder._embed(input_ids, media, {"video": {}}, None, None)

completions = llm.generate(prompts=[{"prompt_embeds": inputs_embeds.squeeze(0)}], sampling_params=SamplingParams(max_tokens=1024))
response = completions[0].outputs[0].text
print("Response: ", response)
```

## Training

### Single Node

Refer to the training scripts in the `examples` directory, for instance, using the command below.

```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-Nodes

Refer to the EasyR1 repo.
[here](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment)

```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

## LongVideo-Reason Dataset

Detailed instructions on the data generation process and model evaluation are available in the [`longvideo-reason`](longvideo-reason/) directory.

## Examples

[Provide images and links to the examples, as in the original README.]

## Contributing

*   Ensure Git is installed.
*   Create a project [fork](https://github.com/NVlabs/Long-RL/fork).
*   Clone the repository.
*   Follow the installation steps.
*   Commit and push changes.
*   Submit a pull request.

## Core Contributors

[List of core contributors]

## Citation

```bibtex
[Provide the BibTeX citations]
```

## Acknowledgements

[List of acknowledgements]