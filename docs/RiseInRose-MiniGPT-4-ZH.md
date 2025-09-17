# MiniGPT-4: Unleash Visual Language Understanding with Advanced LLMs

MiniGPT-4, developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny, revolutionizes visual language understanding by leveraging advanced large language models.  ([Original Repo](https://github.com/RiseInRose/MiniGPT-4-ZH))

**Key Features:**

*   **Visual-Language Alignment:**  MiniGPT-4 aligns frozen visual encoders from BLIP-2 with the powerful Vicuna LLM.
*   **Two-Stage Training:** Employs a two-stage training process, including pre-training and fine-tuning, to optimize performance and usability.
*   **High-Quality Data Generation:**  Utilizes a novel approach to create a high-quality, curated dataset for fine-tuning, enhancing response reliability.
*   **Efficient Fine-tuning:** Achieves significant improvements in performance through an efficient fine-tuning phase, requiring minimal computational resources.
*   **Emerging Capabilities:** Exhibits advanced visual language capabilities, similar to those found in GPT-4.

---

## Quickstart

### 1. Installation

**Prerequisites:** Clone the repository and create a Python environment.

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

### 2. Prepare Vicuna Weights

Follow the instructions [here](PrepareVicuna.md) (translated below) to prepare the Vicuna weights.  Alternatively, download the pre-prepared weights, but note this may raise copyright issues.  The original instructions are:

*Vicuna is an open-source LLM based on LLaMA, offering performance close to ChatGPT.*  *The current version of MiniGPT-4 utilizes Vicuna-13B v1.1.*

*   Download the Vicuna delta weights from [lmsys/vicuna-13b-delta-v1.1](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1):

    ```bash
    git lfs install
    git clone https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
    ```

*   Obtain the original LLaMA-13B weights.  Instructions on how to obtain these from the internet or via a torrent are provided in the original documentation.

*   Convert LLaMA weights to Hugging Face Transformers format.

    *   Install dependencies.
    *   Use the script `src/transformers/models/llama/convert_llama_weights_to_hf.py`.

    ```bash
    python src/transformers/models/llama/convert_llama_weights_to_hf.py \
        --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
    ```

*   Apply the delta weights:

    ```bash
    pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
    python -m fastchat.model.apply_delta --base /path/to/llama-13b-hf/  --target /path/to/save/working/vicuna/weight/  --delta /path/to/vicuna-13b-delta-v0/
    ```

*   The final weights should be in a folder with the following structure:

    ```
    vicuna_weights
    ├── config.json
    ├── generation_config.json
    ├── pytorch_model.bin.index.json
    ├── pytorch_model-00001-of-00003.bin
    ...
    ```

*   Specify the path to the Vicuna weights in the model configuration file (e.g., `minigpt4/configs/models/minigpt4.yaml#L16`).

### 3. Prepare Pre-trained MiniGPT-4 Checkpoint

Download a pre-trained checkpoint:

*   For Vicuna 13B:  [Download Link](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
*   For Vicuna 7B:   [Download Link](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

Then, set the path to the checkpoint in the evaluation configuration file (`eval_configs/minigpt4_eval.yaml#L10`).

### 4. Run the Demo Locally

Run the demo using:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

### 5. Training
MiniGPT-4's training consists of two alignment stages.

**1. Stage 1 Pretraining**

*   Train the model using image-text pairs from the Laion and CC datasets.
*   See [dataset/README_1_STAGE.md](dataset/README_1_STAGE.md) for dataset preparation.
*   Run the following command to start stage 1 training.

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```
*   The checkpoint for stage 1 MiniGPT-4 can be downloaded [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link).

**2. Stage 2 Fine-tuning**

*   Use a small, high-quality image-text pair dataset converted to a dialogue format.
*   See [dataset/README_2_STAGE.md](dataset/README_2_STAGE.md) for dataset preparation.
*   Specify the path of the stage 1 training checkpoint in `train_configs/minigpt4_stage2_finetune.yaml`.
*   Run the following command to start stage 2 fine-tuning:

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

---

## News

*   A pre-trained MiniGPT-4 aligned with Vicuna-7B is now available! The demo can run with as low as 12GB of GPU memory.

---

## Acknowledgements

*   [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [Lavis](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

---

## Citation

If you use MiniGPT-4 in your research, please cite:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models}, 
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

---

## License

This repository is licensed under the [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS), which is licensed under the [BSD 3-Clause License](LICENSE_Lavis.md).