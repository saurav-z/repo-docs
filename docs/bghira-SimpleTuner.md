# SimpleTuner: Train Diffusion Models with Ease 🚀

**SimpleTuner** is your user-friendly toolkit for fine-tuning various diffusion models, offering a straightforward path to customization and creative exploration. [View the original repository](https://github.com/bghira/SimpleTuner)

Key features:

*   **Simplified Training:** Designed for ease of use, with sensible defaults and minimal configuration.
*   **Broad Model Support:** Trains a wide array of models, including HiDream, Flux.1, SDXL, SD3, and more.
*   **Efficient Memory Usage:** Leverages techniques like aspect bucketing, caching, and DeepSpeed integration to minimize VRAM consumption.
*   **Flexible Training Options:** Supports LoRA, LyCORIS, full fine-tuning, ControlNet, and Mixture of Experts.
*   **Advanced Features:** Includes features like EMA, S3 storage support, webhook integration, and Hugging Face Hub integration.

## Core Features

*   Multi-GPU training
*   Image, video, and caption features (embeds) are cached to the hard drive in advance, so that training runs faster and with less memory consumption
*   Aspect bucketing: support for a variety of image/video sizes and aspect ratios, enabling widescreen and portrait training.
*   Refiner LoRA or full u-net training for SDXL
*   Most models are trainable on a 24G GPU, or even down to 16G at lower base resolutions.
    *   LoRA/LyCORIS training for PixArt, SDXL, SD3, and SD 2.x that uses less than 16G VRAM
*   DeepSpeed integration allowing for [training SDXL's full u-net on 12G of VRAM](/documentation/DEEPSPEED.md), albeit very slowly.
*   Quantised NF4/INT8/FP8 LoRA training, using low-precision base model to reduce VRAM consumption.
*   Optional EMA (Exponential moving average) weight network to counteract model overfitting and improve training stability.
*   Train directly from an S3-compatible storage provider, eliminating the requirement for expensive local storage. (Tested with Cloudflare R2 and Wasabi S3)
*   For SDXL, SD 1.x/2.x, and Flux, full or LoRA based [ControlNet model training](/documentation/CONTROLNET.md) (not ControlLite)
*   Training [Mixture of Experts](/documentation/MIXTURE_OF_EXPERTS.md) for lightweight, high-quality diffusion models
*   [Masked loss training](/documentation/DREAMBOOTH.md#masked-loss) for superior convergence and reduced overfitting on any model
*   Strong [prior regularisation](/documentation/DATALOADER.md#is_regularisation_data) training support for LyCORIS models
*   Webhook support for updating eg. Discord channels with your training progress, validations, and errors
*   Integration with the [Hugging Face Hub](https://huggingface.co) for seamless model upload and nice automatically-generated model cards.
    *   Use the [datasets library](/documentation/data_presets/preset_subjects200k.md) ([more info](/documentation/HUGGINGFACE_DATASETS.md)) to load compatible datasets directly from the hub

## Model Support and Quickstarts

SimpleTuner provides specialized support and quickstart guides for the following models:

*   **HiDream:** Custom ControlNet implementation, memory-efficient training, and MoEGate loss augmentation. See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).
*   **Flux.1:**  Accelerated training with `--fuse_qkv_projections`, ControlNet training, and instruct fine-tuning. See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).
*   **Wan Video:** Preliminary text-to-video training integration with LyCORIS, PEFT, and full tuning support. See the [Wan Video Quickstart](/documentation/quickstart/WAN.md) guide.
*   **LTX Video:** Efficiently train LTX Video on less than 16G. See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md) guide.
*   **PixArt Sigma:** Extensive training integration with LyCORIS and full tuning support, including two-stage training. See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide.
*   **NVLabs Sana:** Lightweight and accessible model training with LyCORIS and full tuning. See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md) guide.
*   **Stable Diffusion 3:** Full finetuning and ControlNet training supported.  See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md).
*   **Kwai Kolors:** SDXL-based model with ChatGLM text encoder.
*   **Lumina2:** Flow-matching model with LoRA, Lycoris, and full finetuning support. A [Lumina2 Quickstart](/documentation/quickstart/LUMINA2.md) is available.
*   **Cosmos2 Predict (Image):** Text-to-image variant supported with Lycoris or full-rank tuning. A [Cosmos2 Predict Quickstart](/documentation/quickstart/COSMOS2IMAGE.md) is available.

## Hardware Requirements

Hardware recommendations are model-specific. See below for a general overview:

### NVIDIA

Generally compatible with NVIDIA GPUs 3080 and up.

### AMD

LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.  Xformers not supported.

### Apple

LoRA and full-rank tuning tested on M3 Max with 128GB memory. A 24GB+ machine is likely required.

### HiDream

*   A100-80G (Full tune with DeepSpeed)
*   A100-40G (LoRA, LoKr)
*   3090 24G (LoRA, LoKr)

### Flux.1

*   A100-80G (Full tune with DeepSpeed)
*   A100-40G (LoRA, LoKr)
*   3090 24G (LoRA, LoKr)
*   4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
*   4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### SDXL, 1024px

*   A100-80G (EMA, large batches, LoRA @ insane batch sizes)
*   A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes)
*   A100-40G (EMA@1024px, EMA@768px, EMA@512px, LoRA @ high batch sizes)
*   4090-24G (EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes)
*   4080-12G (LoRA @ low-medium batch sizes)

### Stable Diffusion 2.x, 768px

*   16G or better

##  Resources

*   **Toolkit:** Learn more about the included toolkit in the [toolkit documentation](/toolkit/README.md).
*   **Setup:** Get started with the [installation documentation](/INSTALL.md).
*   **Troubleshooting:**  Enable debug logs by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` and `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` for detailed insights.
*   **Configuration Options:** Explore all available options in [this documentation](/OPTIONS.md).