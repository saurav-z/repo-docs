# SimpleTuner: Unleash Your Creativity with Simplified Diffusion Model Training

SimpleTuner empowers you to easily train and fine-tune various diffusion models, offering a user-friendly experience for both beginners and experienced users.  Explore the power of AI art with SimpleTuner - [check out the original repo](https://github.com/bghira/SimpleTuner).

> ℹ️ This project prioritizes your privacy: no data is sent to third parties unless you explicitly enable options like `report_to`, `push_to_hub`, or configure webhooks manually.

**Key Features:**

*   **Simplified Training:** Get started quickly with intuitive settings and a focus on ease of use.
*   **Versatile Compatibility:** Supports a wide range of image and video datasets, from small collections to massive datasets.
*   **Cutting-Edge Techniques:** Integrates proven features for optimal performance, avoiding unnecessary complexity.
*   **Multi-GPU Training:** Accelerate your training with multi-GPU support.
*   **Advanced Training Methods:** Includes token-wise dropout (TREAD), aspect ratio bucketing, and EMA for enhanced results.
*   **Model Support:**
    *   Flux
    *   Wan Video
    *   LTX Video
    *   PixArt Sigma
    *   NVLabs Sana
    *   Stable Diffusion 2.0/2.1
    *   Stable Diffusion 3.0
    *   Kwai Kolors
    *   Lumina2
    *   Cosmos2 Predict
    *   HiDream
    *   AuraFlow

*   **DeepSpeed Integration:** Leverage DeepSpeed for memory optimization, enabling training on systems with limited VRAM.
*   **Low-Precision Training:** Utilizes quantisation techniques (NF4/INT8/FP8) to reduce VRAM consumption.
*   **S3 Storage Support:** Train directly from S3-compatible storage providers (e.g., Cloudflare R2, Wasabi S3).
*   **ControlNet Training:** Supports ControlNet training for SDXL, SD 1.x/2.x, and Flux.
*   **Hugging Face Hub Integration:** Seamlessly upload and share your models with the Hugging Face Hub.

**Sections:**

*   [Tutorial](/TUTORIAL.md) - Dive into the training process.
*   [Quick Start](/documentation/QUICKSTART.md) - Get up and running fast.
*   [DeepSpeed Document](/documentation/DEEPSPEED.md) - Optimise memory usage.
*   [Distributed Training](/documentation/DISTRIBUTED.md) - Scale training across multiple nodes.
*   [Features](#features) - Detailed overview of SimpleTuner's capabilities.
*   [Hardware Requirements](#hardware-requirements) - Hardware recommendations for optimal performance.
*   [Toolkit](/toolkit/README.md) - Explore the utility toolkit.
*   [Setup](/INSTALL.md) - Installation instructions.
*   [Troubleshooting](#troubleshooting) - Debugging tips and tricks.