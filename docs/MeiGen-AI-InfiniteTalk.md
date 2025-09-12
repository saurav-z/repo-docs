<div align="center">

<p align="center">
  <img src="assets/logo2.jpg" alt="InfinteTalk" width="440"/>
</p>

<h1>InfiniteTalk: Generate Unlimited Talking Videos with Audio</h1>

[Shaoshu Yang*](https://scholar.google.com/citations?user=JrdZbTsAAAAJ&hl=en) · [Zhe Kong*](https://scholar.google.com/citations?user=4X3yLwsAAAAJ&hl=zh-CN) · [Feng Gao*](https://scholar.google.com/citations?user=lFkCeoYAAAAJ) · [Meng Cheng*]() · [Xiangyu Liu*]() · [Yong Zhang](https://yzhang2016.github.io/)<sup>&#9993;</sup> · [Zhuoliang Kang](https://scholar.google.com/citations?user=W1ZXjMkAAAAJ&hl=en)

[Wenhan Luo](https://whluo.github.io/) · [Xunliang Cai](https://openreview.net/profile?id=~Xunliang_Cai1) · [Ran He](https://scholar.google.com/citations?user=ayrg9AUAAAAJ&hl=en)· [Xiaoming Wei](https://scholar.google.com/citations?user=JXV5yrZxj5MC&hl=zh-CN) 

<sup>*</sup>Equal Contribution
<sup>&#9993;</sup>Corresponding Authors

<a href='https://meigen-ai.github.io/InfiniteTalk/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2508.14033'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/MeiGen-AI/InfiniteTalk'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>

</div>

InfiniteTalk revolutionizes video generation, enabling the creation of infinite-length talking videos from audio and images, with unparalleled lip synchronization and identity preservation.  ([See the original repo](https://github.com/MeiGen-AI/InfiniteTalk))

<p align="center">
  <img src="assets/pipeline.png">
</p>

## Key Features

*   **Sparse-frame Video Dubbing:** Synchronizes lips, head movements, body posture, and facial expressions with the audio.
*   **Infinite-Length Generation:** Supports the creation of videos with unlimited durations.
*   **Improved Stability:** Reduces hand/body distortions compared to alternative methods like MultiTalk.
*   **Superior Lip Accuracy:** Achieves enhanced lip synchronization compared to similar models.
*   **Image-to-Video Capabilities:** Transforms a single image and audio into a talking video.

## Community Contributions

*   **Wan2GP:** Integration by [deepbeepmeep](https://github.com/deepbeepmeep) with low VRAM optimization and video editing features.
*   **ComfyUI:** Support by [kijai](https://github.com/kijai).

## Video Demos

### Video-to-video

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/04f15986-8de7-4bb4-8cde-7f7f38244f9f" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/1500f72e-a096-42e5-8b44-f887fa8ae7cb" width="320" controls loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/28f484c2-87dc-4828-a9e7-cb963da92d14" width="320" controls loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/665fabe4-3e24-4008-a0a2-a66e2e57c38b" width="320" controls loop></video>
     </td>
  </tr>
</table>

### Image-to-video

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/7e4a4dad-9666-4896-8684-2acb36aead59" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/bd6da665-f34d-4634-ae94-b4978f92ad3a" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/510e2648-82db-4648-aaf3-6542303dbe22" width="320" controls loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/27bb087b-866a-4300-8a03-3bbb4ce3ddf9" width="320" controls loop></video>
     </td>
     
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/3263c5e1-9f98-4b9b-8688-b3e497460a76" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/5ff3607f-90ec-4eee-b964-9d5ee3028005" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/e504417b-c8c7-4cf0-9afa-da0f3cbf3726" width="320" controls loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/56aac91e-c51f-4d44-b80d-7d115e94ead7" width="320" controls loop></video>
     </td>
     
  </tr>
</table>

## Installation and Quick Start

### 1. Environment Setup

1.  Create a conda environment and install PyTorch and Xformers:

    ```bash
    conda create -n multitalk python=3.10
    conda activate multitalk
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
    ```
2.  Install Flash-attention:

    ```bash
    pip install misaki[en]
    pip install ninja 
    pip install psutil 
    pip install packaging
    pip install wheel
    pip install flash_attn==2.7.4.post1
    ```
3.  Install other dependencies:

    ```bash
    pip install -r requirements.txt
    conda install -c conda-forge librosa
    ```
4.  Install FFmpeg:

    ```bash
    conda install -c conda-forge ffmpeg
    ```
    or
    ```bash
    sudo yum install ffmpeg ffmpeg-devel
    ```

### 2. Model Preparation

1.  Download Models.

    *   Download models using huggingface-cli:
        ```sh
        huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
        huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
        huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
        huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
        ```

### 3. Quick Inference

*   Lip synchronization accuracy: Audio CFG works optimally between 3–5.
*   FusionX: FusionX LoRA exacerbates color shift over 1 minute.
*   V2V generation: Enables unlimited length generation.
*   I2V generation: Generates good results from a single image for up to 1 minute.
*   Quantization model: Use the quantization model to reduce memory usage.

#### 1. Inference (Single GPU)

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res
```

#### 2. Run with 720P

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-720 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_720p
```

#### 3. Run with very low VRAM

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --num_persistent_param_in_dit 0 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_lowvram
```

#### 4. Multi-GPU inference

```bash
GPU_NUM=8
torchrun --nproc_per_node=$GPU_NUM --standalone generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --dit_fsdp --t5_fsdp \
    --ulysses_size=$GPU_NUM \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_multigpu
```

#### 5. Multi-Person animation

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/multi/infinitetalk.safetensors \
    --input_json examples/multi_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --num_persistent_param_in_dit 0 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_multiperson
```

#### 6. Run with FusioniX or Lightx2v

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --lora_dir weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --input_json examples/single_example_image.json \
    --lora_scale 1.0 \
    --size infinitetalk-480 \
    --sample_text_guide_scale 1.0 \
    --sample_audio_guide_scale 2.0 \
    --sample_steps 8 \
    --mode streaming \
    --motion_frame 9 \
    --sample_shift 2 \
    --num_persistent_param_in_dit 0 \
    --save_file infinitetalk_res_lora
```

#### 7. Run with the quantization model (Single GPU)

```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --quant fp8 \
    --quant_dir weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors \
    --motion_frame 9 \
    --num_persistent_param_in_dit 0 \
    --save_file infinitetalk_res_quant
```

#### 8. Run with Gradio

```bash
python app.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --num_persistent_param_in_dit 0 \
    --motion_frame 9 
```
or
```bash
python app.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/multi/infinitetalk.safetensors \
    --num_persistent_param_in_dit 0 \
    --motion_frame 9 
```

## Citation

```
@misc{yang2025infinitetalkaudiodrivenvideogeneration,
      title={InfiniteTalk: Audio-driven Video Generation for Sparse-Frame Video Dubbing}, 
      author={Shaoshu Yang and Zhe Kong and Feng Gao and Meng Cheng and Xiangyu Liu and Yong Zhang and Zhuoliang Kang and Wenhan Luo and Xunliang Cai and Ran He and Xiaoming Wei},
      year={2025},
      eprint={2508.14033},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.14033}, 
}
```

## License

This project is licensed under the Apache 2.0 License.  You are fully accountable for your use of the models, which must not involve sharing any content that violates applicable laws, 
causes harm to individuals or groups, disseminates personal information intended for harm, spreads misinformation, or targets vulnerable populations.