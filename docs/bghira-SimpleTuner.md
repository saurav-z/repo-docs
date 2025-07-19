# SimpleTuner: Train Cutting-Edge Diffusion Models with Ease

SimpleTuner is designed for simplicity and ease of use, empowering you to fine-tune a variety of diffusion models with a focus on clarity and understanding. [Explore the original repository](https://github.com/bghira/SimpleTuner) for further details and contribute to the community.

**Key Features:**

*   ✅ **Simplified Training:** Streamlined setup with good default settings for most use cases.
*   🖼️ **Versatile Support:** Trains on a wide range of image/video datasets, from small to large collections.
*   🚀 **Cutting-Edge Models:** Supports training for popular models like SDXL, Stable Diffusion 3, Flux.1, HiDream, PixArt Sigma, and more.
*   🧠 **Memory Optimization:** Features like aspect bucketing and caching reduce memory consumption.
*   ⚙️ **Advanced Techniques:** Integrates DeepSpeed for large models, quantization for VRAM savings, EMA, and more.
*   🌐 **Integration:** Seamlessly integrates with Hugging Face Hub and supports S3 storage.
*   🛠️ **ControlNet Support:** Train ControlNet models (full or LoRA) for SDXL, SD 1.x/2.x, and Flux.
*   📊 **Training Toolkit:** Provides useful tools for fine-tuning your models.
*   📦 **Easy to Set up:** See the installation documentation for the specifics.
*   🛡️ **Privacy Focused:** No data sent to third parties except via opt-in flags or manual webhook configuration.

**Ready to get started?**

*   Refer to the [tutorial](/TUTORIAL.md) for comprehensive information.
*   Use the [Quick Start](/documentation/QUICKSTART.md) for a faster introduction.
*   Learn how to use [DeepSpeed](/documentation/DEEPSPEED.md) for memory-constrained systems.
*   See [this guide](/documentation/DISTRIBUTED.md) to set up multi-node training.

**Model Support Highlights:**

*   **HiDream:** Custom ControlNet, memory-efficient training for NVIDIA GPUs.
*   **Flux.1:** Training support for both Dev and Full.
*   **Wan Video:** Text-to-video training (14B and 1.3B).
*   **LTX Video:** Efficient training on less than 16G VRAM.
*   **PixArt Sigma:** Extensive integration with the 600M & 900M models.
*   **NVLabs Sana:** A lightweight, fast model for accessible training.
*   **Stable Diffusion 3:** LoRA, full finetuning, and ControlNet training are supported.
*   **Kwai Kolors:** SDXL-based model with ChatGLM 6B text encoder
*   **Lumina2:** LoRA, Lycoris, and full finetuning are supported
*   **Cosmos2 Predict (Image)** Lycoris or full-rank tuning are supported

**Hardware Requirements:**

*   **NVIDIA:** 3080 and up recommended.
*   **AMD:** LoRA and full-rank tuning verified on 7900 XTX 24GB and MI300X.
*   **Apple:** M3 Max (128GB) is recommended
*   Specific requirements for models like HiDream, Flux.1, SDXL, and others are detailed in the original README.

**Further Information:**

*   Explore the [toolkit documentation](/toolkit/README.md) for tools.
*   Read the [installation documentation](/INSTALL.md) for setup instructions.
*   For troubleshooting, use the environment variables (see original README)
*   See the [options documentation](/OPTIONS.md) for a comprehensive list of options.