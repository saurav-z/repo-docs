# $\pi^3$: Revolutionizing Visual Geometry Learning with Scalable, Permutation-Equivariant Networks

[Link to Original Repo](https://github.com/yyfz/Pi3)

$\pi^3$ (**Pi-Cubed**) introduces a novel neural network architecture for robust and scalable visual geometry reconstruction, eliminating the need for a fixed reference view.

[![Paper](https://img.shields.io/badge/Paper-00AEEF?style=plastic&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2507.13347)
[![Project Page](https://img.shields.io/badge/Project%20Page-F78100?style=plastic&logo=google-chrome&logoColor=white)](https://yyfz.github.io/pi3/)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/yyfz233/Pi3)

<div align="center">
    <a href="[PROJECT_PAGE_LINK_HERE]">
        <img src="assets/main.png" width="90%" alt="Pi3 Reconstruction Example">
    </a>
    <p>
        <i>&pi;³ reconstructs visual geometry without a fixed reference view, achieving robust, state-of-the-art performance.</i>
    </p>
</div>

**Key Features:**

*   **Permutation-Equivariant Architecture:** Enables robust reconstruction from unordered sets of images, eliminating the need for a fixed reference frame.
*   **Scalable and Robust:**  Inherently handles different input orderings, leading to superior performance and stability.
*   **State-of-the-Art Performance:** Achieves leading results in camera pose estimation, depth estimation, and dense point map estimation.
*   **Hugging Face Demo:**  Try out $\pi^3$ with an interactive demo.
*   **Easy-to-Use:**  Includes a quick start guide with example code and command-line inference.

## 📣 What's New
*   **[July 29, 2025]** 📈 Evaluation code released! See `evaluation` branch.
*   **[July 16, 2025]** 🚀 Hugging Face Demo and inference code are released!

## 🚀 Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/yyfz/Pi3.git
cd Pi3
pip install -r requirements.txt
```

### 2. Run Inference from Command Line

```bash
# Run with default example video
python example.py

# Run on your own data (image folder or .mp4 file)
python example.py --data_path <path/to/your/images_dir_or_video.mp4>
```

**Optional Arguments:**

*   `--data_path`: Path to input image directory or video file. (Default: `examples/skating.mp4`)
*   `--save_path`: Path to save output `.ply` point cloud. (Default: `examples/result.ply`)
*   `--interval`: Frame sampling interval. (Default: `1` for images, `10` for video)
*   `--ckpt`: Path to custom model checkpoint.
*   `--device`: Device to run inference on. (Default: `cuda`)

### 3. Run with Gradio Demo

```bash
# Install demo-specific requirements
pip install -r requirements_demo.txt

# Launch the demo
python demo_gradio.py
```

## 🛠️ Detailed Usage

### Model Input & Output

*   **Input:** `torch.Tensor` of shape $B \times N \times 3 \times H \times W$ (pixel values in `[0, 1]`).
*   **Output:** `dict` containing:
    *   `points`: Global point cloud ($B \times N \times H \times W \times 3$).
    *   `local_points`: Per-view local point maps ($B \times N \times H \times W \times 3$).
    *   `conf`: Confidence scores for local points (values in `[0, 1]`, higher is better) ($B \times N \times H \times W \times 1$).
    *   `camera_poses`: Camera-to-world transformation matrices (4x4 OpenCV format) ($B \times N \times 4 \times 4$).

### Example Code Snippet

```python
import torch
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
imgs = load_images_as_tensor('path/to/your/data', interval=10).to(device)

print("Running model inference...")
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=dtype):
        results = model(imgs[None])

print("Reconstruction complete!")
# Access outputs: results['points'], results['camera_poses'] and results['local_points'].
```

## 🙏 Acknowledgements

Our work builds upon these open-source projects:

*   [DUSt3R](https://github.com/naver/dust3r)
*   [CUT3R](https://github.com/CUT3R/CUT3R)
*   [VGGT](https://github.com/facebookresearch/vggt)

## 📜 Citation

```bibtex
@misc{wang2025pi3,
      title={$\pi^3$: Scalable Permutation-Equivariant Visual Geometry Learning}, 
      author={Yifan Wang and Jianjun Zhou and Haoyi Zhu and Wenzheng Chang and Yang Zhou and Zizun Li and Junyi Chen and Jiangmiao Pang and Chunhua Shen and Tong He},
      year={2025},
      eprint={2507.13347},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.13347}, 
}
```

## 📄 License

Licensed under the 2-clause BSD License for academic use.  Contact authors for commercial use.