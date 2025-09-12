## MiniGPT-4: Unlock Visual Language Understanding with Advanced AI

**MiniGPT-4 enhances visual language understanding using cutting-edge large language models, letting you chat about images like never before!**  Explore the original repository [here](https://github.com/RiseInRose/MiniGPT-4-ZH).

**Authors:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny (King Abdullah University of Science and Technology)

**Key Features:**

*   **Image-to-Text Generation:** Generate detailed descriptions of images.
*   **Visual Question Answering:** Answer questions based on image content.
*   **Interactive Chat:** Engage in conversations about images, exploring their details and context.
*   **Two-Stage Training:** Leverages a two-stage training process for enhanced performance.
*   **User-Friendly:** Designed for ease of use, with a focus on generating reliable and coherent responses.
*   **Efficient Fine-tuning:**  Second-stage fine-tuning requires minimal computational resources.
*   **Integration of Existing Models:** Utilizes frozen visual encoders from BLIP-2 and the Vicuna LLM.

**Online Demo:**
Try out the online demo and see MiniGPT-4 in action:
[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

**Resources:**

*   [Project Page](https://minigpt-4.github.io)
*   [Paper (PDF)](MiniGPT_4.pdf)
*   [Hugging Face Spaces](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
*   [Hugging Face Model](https://huggingface.co/Vision-CAIR/MiniGPT-4)
*   [Google Colab](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
*   [YouTube Video](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

---

**News:**

*   Pre-trained MiniGPT-4 with Vicuna-7B alignment is now available!  Demo GPU memory consumption can be as low as 12GB.

---

**Getting Started:**

### Installation

**1.  Prepare Code and Environment:**

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2.  Prepare Pre-trained Vicuna Weights:**

*   Follow the instructions [here](PrepareVicuna.md) to prepare the Vicuna weights. *Note:  You will need to download the LLAMA weights, and use the instructions in the original README to obtain them.*

**3.  Prepare Pre-trained MiniGPT-4 Checkpoint:**

*   Download pre-trained checkpoints based on your Vicuna model.

    | Checkpoint Aligned with Vicuna 13B                                | Checkpoint Aligned with Vicuna 7B                                |
    | :-----------------------------------------------------------------: | :-------------------------------------------------------------: |
    | [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) |

*   Set the path to the pre-trained checkpoint in `eval_configs/minigpt4_eval.yaml` (line 11).

### Launching the Demo Locally:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

### Training

MiniGPT-4 training involves two alignment stages.

**1. Stage 1 Pre-training:**
Follow the  [Stage 1 Dataset preparation instructions](dataset/README_1_STAGE.md). To begin the pre-training stage run this command.
```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```
  First stage MiniGPT-4 checkpoints are available [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link).

**2. Stage 2 Fine-tuning:**
Follow the [Stage 2 dataset preperation instructions](dataset/README_2_STAGE.md). To begin the fine-tuning stage, set the path to the Stage 1 checkpoint in train_configs/minigpt4_stage2_finetune.yaml (there you can also specify the output path) and run the following command.
```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```
---
**Acknowledgments:**

*   [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [Lavis](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

**Citation:**

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

---

**License:**  [BSD 3-Clause License](LICENSE.md).  Code is based on [Lavis](https://github.com/salesforce/LAVIS) ([BSD 3-Clause License](LICENSE_Lavis.md)).