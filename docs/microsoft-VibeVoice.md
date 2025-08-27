# VibeVoice: Generate Expressive, Long-Form Conversational Audio with AI

**VibeVoice is a cutting-edge text-to-speech model that can transform text into natural-sounding, multi-speaker conversations like podcasts, pushing the boundaries of audio generation.** [Learn more at the original repo](https://github.com/microsoft/VibeVoice).

## Key Features:

*   **Long-Form Audio Generation:** Generate audio up to 90 minutes in length.
*   **Multi-Speaker Support:** Supports up to 4 distinct speakers in a single conversation.
*   **High-Fidelity Speech:** Employs continuous speech tokenizers to preserve audio quality.
*   **LLM-Powered Dialogue Understanding:** Leverages a Large Language Model (LLM) to understand textual context and conversation flow.
*   **Open Source:**  The VibeVoice-7B-Preview weights are now available on Hugging Face!

## Getting Started

### Models

| Model                     | Context Length | Generation Length | Weight                                                                     |
| ------------------------- | -------------- | ----------------- | -------------------------------------------------------------------------- |
| VibeVoice-0.5B-Streaming  | -              | -                 | On the way                                                                  |
| VibeVoice-1.5B            | 64K            | ~90 min          | [Hugging Face Link](https://huggingface.co/microsoft/VibeVoice-1.5B)         |
| VibeVoice-7B-Preview      | 32K            | ~45 min          | [Hugging Face Link](https://huggingface.co/WestZhang/VibeVoice-Large-pt)     |

### Installation

1.  **Set up your environment:**  It is recommended to use the NVIDIA Deep Learning Container to manage your CUDA environment. Follow the instructions in the original README to launch a compatible container.
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/microsoft/VibeVoice.git
    cd VibeVoice/
    ```
3.  **Install the package:**
    ```bash
    pip install -e .
    ```

### Usage

#### Gradio Demo

```bash
apt update && apt install ffmpeg -y # for demo

# For 1.5B model
python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B --share

# For 7B model
python demo/gradio_demo.py --model_path WestZhang/VibeVoice-Large-pt --share
```

#### Inference from Files

```bash
# 1 speaker
python demo/inference_from_file.py --model_path WestZhang/VibeVoice-Large-pt --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice

# 2 or more speakers
python demo/inference_from_file.py --model_path WestZhang/VibeVoice-Large-pt --txt_path demo/text_examples/2p_music.txt --speaker_names Alice Yunfan
```

## Examples
*   **Cross-Lingual:** [Demo](https://github.com/user-attachments/assets/838d8ad9-a201-4dde-bb45-8cd3f59ce722)
*   **Spontaneous Singing:** [Demo](https://github.com/user-attachments/assets/6f27a8a5-0c60-4f57-87f3-7dea2e11c730)
*   **Long Conversation with 4 people:** [Demo](https://github.com/user-attachments/assets/a357c4b6-9768-495c-a576-1618f6275727)

For more examples, see the [Project Page](https://microsoft.github.io/VibeVoice) and try your own samples at [Demo](https://aka.ms/VibeVoice-Demo).

## Frequently Asked Questions (FAQ)

*   **Q1: Is this a pretrained model?**  
    **A:** Yes, VibeVoice is a pretrained model.
*   **Q2: Randomly trigger Sounds / Music / BGM.**  
    **A:** Background music is an emergent feature, not directly controllable by the user. This can be context-aware.
*   **Q3: Text normalization?**  
    **A:** The model does not perform text normalization.
*   **Q4: Singing Capability.**  
    **A:** Singing is an emergent ability, not explicitly trained for.
*   **Q5: Some Chinese pronunciation errors.**  
    **A:** The volume of Chinese data in our training set is significantly smaller than the English data.

## Risks and Limitations

*   **Deepfakes and Disinformation:** Potential for misuse in creating synthetic audio.
*   **Language Support:** Primarily supports English and Chinese.
*   **Non-Speech Audio:** Does not handle background noise, music, or sound effects explicitly.
*   **Overlapping Speech:** Does not model overlapping speech.

**Use responsibly. This model is intended for research and development purposes only.**