# MiniGPT-4: Unleash the Power of Visual Language Understanding with Large Language Models

**MiniGPT-4 enables you to have in-depth conversations about images by combining a visual encoder with the powerful Vicuna language model.** Developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny, this project from King Abdullah University of Science and Technology (KAUST) offers a novel approach to understanding and interacting with visual information.  [Check out the original repository](https://github.com/RiseInRose/MiniGPT-4-ZH).

## Key Features:

*   **Advanced Visual Language Understanding:** Leverages a frozen visual encoder (from BLIP-2) and the powerful Vicuna LLM to understand and generate text descriptions and dialogues about images.
*   **Two-Stage Training:**  Employs a two-stage training process: a pre-training phase with 5 million image-text pairs, followed by a fine-tuning stage on a high-quality, curated dataset, significantly improving the model's generation capabilities and usability.
*   **Enhanced Usability:** MiniGPT-4 is designed for ease of use, with the fine-tuning phase making it more reliable and user-friendly.
*   **Open-Source & Accessible:** Built upon open-source projects like BLIP-2, Lavis, and Vicuna, facilitating access and community contribution.
*   **Efficient Fine-Tuning:** The second stage of fine-tuning is computationally efficient, requiring only a single A100 GPU for approximately 7 minutes.

## Quick Start:

### Online Demo
Experience MiniGPT-4 firsthand!  Interact with the model and explore its capabilities:
[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

### Installation and Usage (Simplified)

**1. Set Up Your Environment:**

   *   Clone the repository:
      ```bash
      git clone https://github.com/Vision-CAIR/MiniGPT-4.git
      cd MiniGPT-4
      ```
   *   Create and activate a Conda environment:
      ```bash
      conda env create -f environment.yml
      conda activate minigpt4
      ```

**2. Prepare the Vicuna Weights:**

   *  **Option 1 (Recommended):** The provided documentation gives instructions on how to download and convert the Vicuna-13B weights and is more comprehensive.
   *  **Option 2:** Refer to the original documentation for the full details.

**3. Prepare the MiniGPT-4 Checkpoint:**
    * Download pre-trained checkpoints.
    * For Vicuna 13B: [Download Link](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
    * For Vicuna 7B: [Download Link](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

**4. Run the Demo:**

   *   Execute the demo script:
      ```bash
      python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
      ```

**Important Notes:**

*   The original repository includes detailed instructions regarding weight preparation and potential troubleshooting (e.g. issues with windows deployment).
*   Model clipping is mentioned but not suggested unless you are experience.
*   Modify the configuration file to run on less resources.

## Training:

MiniGPT-4 is trained in two stages.
*   **Stage 1: Pretraining**:  Uses image-text pairs to align the visual and language models. Instructions for dataset preparation and the training command are available in the original README.  A pre-trained checkpoint is available for download.
*   **Stage 2: Fine-tuning**:  Uses a smaller, high-quality dataset in a dialogue format to further align MiniGPT-4, improving generation quality and usability. Instructions are provided.

## Community & Resources:

*   **Project Page:** [https://minigpt-4.github.io](https://minigpt-4.github.io)
*   **Paper:** [![Paper-PDF-red](https://img.shields.io/badge/Paper-PDF-red)](MiniGPT_4.pdf)
*   **Hugging Face Spaces:**  [![Hugging Face-Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
*   **Hugging Face Model:** [![Hugging Face-Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Vision-CAIR/MiniGPT-4)
*   **Colab:** [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
*   **YouTube Demo:** [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

##  Citation:

If you use MiniGPT-4 in your research, please cite the following:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models}, 
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License

This project is licensed under the [BSD 3-Clause License](LICENSE.md). Code based on Lavis uses the [BSD 3-Clause License](LICENSE_Lavis.md).