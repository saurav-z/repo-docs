# $\pi^3$: Revolutionizing Visual Geometry Learning with Permutation Equivariance

$\pi^3$ is a cutting-edge model that reconstructs visual geometry without relying on a fixed reference view, achieving state-of-the-art performance in 3D reconstruction tasks.  Explore the details and code at the [original repository](https://github.com/yyfz/Pi3).

**Key Features:**

*   **Permutation-Equivariant Architecture:** Enables robust and scalable 3D reconstruction from unordered image sets, eliminating the need for a fixed reference view.
*   **State-of-the-Art Performance:** Achieves top results in camera pose estimation, depth estimation, and dense point map estimation.
*   **Scalable & Robust:**  Designed to handle varying input orders and image sequences without compromising accuracy.
*   **Hugging Face Demo:** Easily test and visualize the model's capabilities with an interactive demo.
*   **Open Source:** Available under the 2-clause BSD License for academic use.

##  Updates
*   **[July 29, 2025]** 📈 Evaluation code released! Explore the `evaluation` branch.
*   **[July 16, 2025]** 🚀 Hugging Face Demo and inference code are now available!

## ✨ Overview

$\pi^3$ (Pi-Cubed) introduces a novel feed-forward neural network architecture to address limitations of traditional visual geometry reconstruction methods. Unlike conventional approaches that require a fixed reference view, which can be unstable, $\pi^3$ embraces a fully **permutation-equivariant** design.  This innovative approach empowers the model to directly predict affine-invariant camera poses and scale-invariant local point maps from an unordered set of images. This inherent design choice ensures the model is robust to the order of input images and allows for unmatched scalability. The model achieves **state-of-the-art performance** on camera pose estimation, monocular/video depth estimation, and dense point map estimation.

<div align="center">
    <a href="[PROJECT_PAGE_LINK_HERE]">
        <img src="assets/main.png" width="90%">
    </a>
    <p>
        <i>&pi;³ reconstructs visual geometry without a fixed reference view, achieving robust, state-of-the-art performance.</i>
    </p>
</div>

## 🚀 Quick Start

Get started with $\pi^3$ in just a few steps:

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/yyfz/Pi3.git
cd Pi3
pip install -r requirements.txt
```

### 2. Run Inference

Quickly test the model using the example inference script:

```bash
# Run with default example video
python example.py

# Run on your own data (image folder or .mp4 file)
python example.py --data_path <path/to/your/images_dir_or_video.mp4>
```

**Optional Arguments:**

*   `--data_path`: Path to the input image directory or a video file (default: `examples/skating.mp4`).
*   `--save_path`: Path to save the output `.ply` point cloud (default: `examples/result.ply`).
*   `--interval`: Frame sampling interval (default: `1` for images, `10` for video).
*   `--ckpt`: Path to a custom model checkpoint file.  If slow, download the model checkpoint manually from [here](https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors)
*   `--device`: Device to run inference on (default: `cuda`).

### 3. Run with Gradio Demo

Launch a local Gradio demo for an interactive experience:

```bash
# Install demo-specific requirements
pip install -r requirements_demo.txt

# Launch the demo
python demo_gradio.py
```

## 🛠️ Detailed Usage

### Model Input & Output

Learn about input and output data structures to use the model efficiently:

*   **Input:** A `torch.Tensor` of shape $B \times N \times 3 \times H \times W$ (batch size, number of images, channels, height, width) with pixel values in the range `[0, 1]`.
*   **Output:** A `dict` containing the reconstructed geometry:
    *   `points`: Global point cloud (shape: $B \times N \times H \times W \times 3$).
    *   `local_points`: Per-view local point maps (shape: $B \times N \times H \times W \times 3$).
    *   `conf`: Confidence scores for local points (values in `[0, 1]`, higher is better) (shape: $B \times N \times H \times W \times 1$).
    *   `camera_poses`: Camera-to-world transformation matrices (4x4 in OpenCV format) (shape: $B \times N \times 4 \times 4$).

### Example Code Snippet

A minimal example of running the model on a batch of images:

```python
import torch
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor # Assuming you have a helper function

# --- Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
# or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`

# --- Load Data ---
# Load a sequence of N images into a tensor
# imgs shape: (N, 3, H, W).
# imgs value: [0, 1]
imgs = load_images_as_tensor('path/to/your/data', interval=10).to(device)

# --- Inference ---
print("Running model inference...")
# Use mixed precision for better performance on compatible GPUs
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=dtype):
        # Add a batch dimension -> (1, N, 3, H, W)
        results = model(imgs[None])

print("Reconstruction complete!")
# Access outputs: results['points'], results['camera_poses'] and results['local_points'].
```

## 🙏 Acknowledgements

We acknowledge the contributions of the authors of the following open-source projects:

*   [DUSt3R](https://github.com/naver/dust3r)
*   [CUT3R](https://github.com/CUT3R/CUT3R)
*   [VGGT](https://github.com/facebookresearch/vggt)

## 📜 Citation

If you use this work, please cite it as:

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

This project is licensed under the 2-clause BSD License for academic use.  Please review the [LICENSE](./LICENSE) file for full details.  For commercial use, please contact the authors.