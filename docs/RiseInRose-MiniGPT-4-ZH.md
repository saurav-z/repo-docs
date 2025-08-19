# MiniGPT-4: Unleash Visual Language Understanding with Advanced LLMs

**Effortlessly chat with images and unlock detailed insights using MiniGPT-4, a powerful vision-language model.**  **(Original Repository: [https://github.com/RiseInRose/MiniGPT-4-ZH](https://github.com/RiseInRose/MiniGPT-4-ZH))**

**Authors:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny (King Abdullah University of Science and Technology).

## Key Features

*   **Image-Based Chat:** Interact with the model by simply uploading an image and asking questions.
*   **Enhanced Visual Language Understanding:** Leverages advanced Large Language Models (LLMs) to interpret and describe images with remarkable accuracy.
*   **Two-Stage Training:** Utilizes a two-stage training process for optimized performance and enhanced usability.
*   **Open Source:** Based on the popular Vicuna LLM, allowing for customization and experimentation.
*   **Easy to Use:**  Local demo and Colab notebook available for easy experimentation and deployment.

## Quick Start

### 1. Installation

**1.1. Clone the Repository and Create Environment:**

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**1.2. Prepare Vicuna Weights:**

**Important Note:** Due to licensing, Vicuna weights themselves are not directly distributed with the repository.  You'll need to obtain these separately.  Follow the instructions below.

#### 1.2.1. Download Vicuna Delta Weights (v1.1):

```bash
git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
```

#### 1.2.2. Obtain LLaMA-13B Original Weights:

Download the LLaMA-13B weights from the specified resources (see original README for links). You may need to fill a form to gain access.

#### 1.2.3. Convert LLaMA Weights to Hugging Face Format:

This step requires the `transformers` library and is essential for compatibility.

```bash
# Install environment dependencies (recommended to use original pip source)
pip install transformers[sentencepiece]
```

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

**Note:** Replace `/path/to/downloaded/llama/weights` and `/output/path` with your actual file paths.  Refer to original README for potential error resolutions.

#### 1.2.4. Create Final Working Weights:

```bash
pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
```

```bash
python -m fastchat.model.apply_delta --base /path/to/llama-13b-hf/  --target /path/to/save/working/vicuna/weight/  --delta /path/to/vicuna-13b-delta-v0/
```

**Note:** Replace the paths in the command above with the locations of the base LLaMA weights, the directory to save your final weights, and the Vicuna delta weights, respectively.

#### 1.2.5. Configure Weights Path:

Specify the path to your generated Vicuna weights in `minigpt4/configs/models/minigpt4.yaml` (line 16).

### 1.3. Prepare Pre-trained MiniGPT-4 Checkpoint

Download the pre-trained checkpoint aligned with your Vicuna model (13B or 7B) from the provided links in the original README. Then, set the checkpoint path in `eval_configs/minigpt4_eval.yaml` (line 11).

### 2. Run the Demo

Launch the local demo using:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

**Note:** The demo requires approximately 23GB of GPU memory for Vicuna 13B, or 11.5GB for Vicuna 7B.

## Training

MiniGPT-4 is trained in two stages:

*   **Stage 1: Pre-training:** Train the model using image-text pairs from Laion and CC datasets to align visual and language models.  See `dataset/README_1_STAGE.md` for dataset preparation.
    Run with:
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```

*   **Stage 2: Fine-tuning:** Fine-tune the model using a small, high-quality dataset in a dialogue format for improved performance. See `dataset/README_2_STAGE.md` for dataset preparation.  Specify the Stage 1 checkpoint path in `train_configs/minigpt4_stage2_finetune.yaml`.
    Run with:
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

## Resources

*   **Online Demo:**  [https://minigpt-4.github.io](https://minigpt-4.github.io)
*   **Project Page:** [https://minigpt-4.github.io](https://minigpt-4.github.io)
*   **Paper:**  [MiniGPT_4.pdf](MiniGPT_4.pdf)
*   **Hugging Face Spaces:**  [https://huggingface.co/spaces/Vision-CAIR/minigpt4](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
*   **Hugging Face Model:** [https://huggingface.co/Vision-CAIR/MiniGPT-4](https://huggingface.co/Vision-CAIR/MiniGPT-4)
*   **Colab:** [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
*   **YouTube:**  [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## Acknowledgements

*   **BLIP-2:** (Model Architecture)
*   **Lavis:** (Framework)
*   **Vicuna:** (LLM Foundation)

## Citation

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models}, 
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License

This project is licensed under the [BSD 3-Clause License](LICENSE.md).  Code is based on [Lavis](https://github.com/salesforce/LAVIS) (BSD 3-Clause License [LICENSE_Lavis.md]).

---

**For the original repository and further details, visit: [https://github.com/RiseInRose/MiniGPT-4-ZH](https://github.com/RiseInRose/MiniGPT-4-ZH)**