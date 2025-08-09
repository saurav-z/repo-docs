<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
<p>
<p align="center">&nbsp&nbsp💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp&nbsp |
           &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog</a> &nbsp&nbsp 
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
<p>

# Qwen-Image: Unleash Stunning Visual Creation with Advanced Image Generation and Editing

**Qwen-Image is a cutting-edge 20B MMDiT image foundation model, revolutionizing image generation and editing with unparalleled text rendering capabilities.** ([Original Repo](https://github.com/QwenLM/Qwen-Image))

## Key Features

*   **Exceptional Text Rendering:** Generate images with precise and clear text, supporting both English and Chinese with impressive accuracy.
*   **Versatile Image Generation:** Create diverse art styles, from photorealistic scenes to artistic paintings and designs.
*   **Advanced Image Editing:** Perform sophisticated edits including style transfer, object manipulation, and human pose adjustments.
*   **Image Understanding Capabilities:** Benefit from integrated object detection, semantic segmentation, and more.
*   **Multi-GPU API Server:** Deploy Qwen-Image locally with an easy-to-use Gradio interface.

## News and Updates

*   **2025.08.05:** Qwen-Image is now natively supported in ComfyUI, see [Qwen-Image in ComfyUI: New Era of Text Generation in Images!](https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of)
*   **2025.08.05:** Qwen-Image is now on Qwen Chat. Click [Qwen Chat](https://chat.qwen.ai/) and choose "Image Generation".
*   **2025.08.05:** Technical Report released on [Arxiv](https://arxiv.org/abs/2508.02324)!
*   **2025.08.04:** Qwen-Image weights are now available on [Hugging Face](https://huggingface.co/Qwen/Qwen-Image) and [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image)!
*   **2025.08.04:** Official Blog Release: Check the [Blog](https://qwenlm.github.io/blog/qwen-image) for more details.

## Quick Start

1.  **Prerequisites:** Ensure you have `transformers>=4.51.3` installed.

2.  **Install Diffusers:**
    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

3.  **Code Example:**

    ```python
    from diffusers import DiffusionPipeline
    import torch

    model_name = "Qwen/Qwen-Image"

    # Load the pipeline... (rest of the code from original README)
    ```

## Show Cases

Qwen-Image excels in text rendering, artistic image generation, and advanced image editing.

### Text Rendering
Showcases impressive text rendering capabilities for various languages, ensuring accurate reproduction.
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s1.jpg#center)

### General Image Generation
Showcases image generation with a wide variety of art styles.
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s2.jpg#center)

### Image Editing
Showcases advanced editing functions like style transfer, object insertion, detail enhancement and human pose manipulation.
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s3.jpg#center)

### Image Understanding
Showcases image understanding tasks, including object detection and depth estimation.
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s4.jpg#center)

## Advanced Usage

### Prompt Enhancement

Enhance prompts using the official Prompt Enhancement Tool powered by Qwen-Plus.

```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

### Multi-GPU API Server

Deploy Qwen-Image locally with a Gradio-based web interface.

**Configuration:**
```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```

**Run the Server:**

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py
```

## AI Arena

Evaluate Qwen-Image's performance on the open benchmarking platform, [AI Arena](https://aiarena.alibaba-inc.com).

![AI Arena](assets/figure_aiarena_website.png)

View the latest rankings on the [AI Arena Leaderboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image)

## Community Support

*   **Hugging Face:**  Supported on Diffusers since day 0.
*   **Modelscope:** Support for layer-by-layer offload, FP8 quantization, LoRA/full training.
*   **WaveSpeedAI:** [Model Page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image)
*   **LiblibAI:** [Community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c)

## License

Qwen-Image is licensed under Apache 2.0.

## Citation

```bibtex
@misc{wu2025qwenimagetechnicalreport,
      title={Qwen-Image Technical Report}, 
      author={Chenfei Wu and Jiahao Li and Jingren Zhou and Junyang Lin and Kaiyuan Gao and Kun Yan and Sheng-ming Yin and Shuai Bai and Xiao Xu and Yilei Chen and Yuxiang Chen and Zecheng Tang and Zekai Zhang and Zhengyi Wang and An Yang and Bowen Yu and Chen Cheng and Dayiheng Liu and Deqing Li and Hang Zhang and Hao Meng and Hu Wei and Jingyuan Ni and Kai Chen and Kuan Cao and Liang Peng and Lin Qu and Minggang Wu and Peng Wang and Shuting Yu and Tingkun Wen and Wensen Feng and Xiaoxiao Xu and Yi Wang and Yichang Zhang and Yongqiang Zhu and Yujia Wu and Yuxuan Cai and Zenan Liu},
      year={2025},
      eprint={2508.02324},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.02324}, 
}
```

## Contact and Join Us

Join our [Discord](https://discord.gg/z3GAxXZ9Ce) or WeChat groups (scan QR code) for discussions and collaboration.

Contribute via issues and pull requests on [GitHub](https://github.com/QwenLM/Qwen-Image).

For FTEs and research intern positions, reach out to fulai.hr@alibaba-inc.com.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)