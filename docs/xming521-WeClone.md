# WeClone: Create Your Digital Avatar from Chat History

**Turn your chat history into a personalized digital avatar with WeClone!**  This project empowers you to create a digital representation of yourself, leveraging the power of Large Language Models (LLMs) and your own conversation data.

[<img src="https://github.com/xming521/WeClone/blob/master/README.md/assets/download.png?raw=true" alt="Demo Image" width="200">](https://github.com/xming521/WeClone) 
<br>
[See the original repository on GitHub](https://github.com/xming521/WeClone)

---

## Key Features

*   **End-to-End Solution:** From data extraction and preprocessing to model training and deployment, WeClone offers a complete workflow.
*   **Personalized LLM Fine-tuning:**  Fine-tune LLMs using your chat history (including images), injecting your unique communication style.
*   **Multi-Platform Integration:**  Integrate with Telegram, and more platforms coming soon.
*   **Privacy-Focused:**  Includes privacy information filtering and localized fine-tuning for data security and control.

---

## Supported Platforms

| Platform  | Text | Images |  Forward |  Animated Emojis/Stickers |  Location |  Files |
|-----------|------|--------|-------|-----------------|----------|----------|
| **WeChat** | ✅   | ✅     |❌| ❌  | ❌ | ❌ |
| **Telegram**| ✅   | ✅     |✅ | ⚠️Convert to Emoji | ✅ | ❌ |

---

## Getting Started

### Hardware Requirements
*   **Model Dependent:**  Uses Qwen2.5-VL-7B-Instruct by default, but supports others via LLaMA Factory.

*   **Estimated VRAM Requirements:**  (These are estimates, see the original README for more details). Consider using a more efficient method like QLoRA to save resources.
    | Method                          | Precision |   7B  |  14B  |  30B  |   70B  |   `x`B  |
    | ------------------------------- | --------- | ----- | ----- | ----- | ------ | ------- |
    | QLoRA (4-bit)                   |     4     |   6GB |  12GB |  24GB |   48GB | `x/2`GB |

### Environment Setup

1.  **CUDA Installation:**  (If not already installed, requires version 12.6 or above).
2.  **Dependency Management (Recommended):** Use [uv](https://docs.astral.sh/uv/) to create and activate a Python environment:
    ```bash
    git clone https://github.com/xming521/WeClone.git && cd WeClone
    uv venv .venv --python=3.10
    source .venv/bin/activate # windows .venv\Scripts\activate
    uv pip install --group main -e .
    ```
3.  **Configuration:** Copy and rename the template to `settings.jsonc` and customize.
    ```bash
    cp examples/tg.template.jsonc settings.jsonc
    ```
4.  **CUDA Verification (Optional):**  Test CUDA configuration.
    ```bash
    python -c "import torch; print('CUDA Available:', torch.cuda.is_available());"
    ```
5.  **FlashAttention (Optional):** Install for accelerated training/inference: `uv pip install flash-attn --no-build-isolation`.

### Model Download

Download your chosen model from Hugging Face or use the following command:

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct models/Qwen2.5-VL-7B-Instruct
```

### Data Preparation

*   **Export Chat Records:**  Use [Telegram Desktop](https://desktop.telegram.org/) to export your chat history (Photos, JSON format).
*   **Directory Structure:**  Place the exported `ChatExport_*` folders into the `./dataset/telegram` directory.

### Data Preprocessing

*   **Configuration:**  Modify `language`, `platform`, `include_type`, and `telegram_args.my_id` (if using Telegram) in `settings.jsonc`.
*   **Privacy Filtering:**  Uses Microsoft Presidio for default PII removal.  Add custom filters in `blocked_words` in `settings.jsonc`.
*   **Run the Preprocessing:**
    ```bash
    weclone-cli make-dataset
    ```
    *   See the documentation for detailed parameter options.

### Fine-tuning and Inference

*   **Configuration:**  Adjust `model_name_or_path`, `template`, `lora_target`, `per_device_train_batch_size`, `gradient_accumulation_steps`, and training parameters (e.g., `num_train_epochs`, `lora_rank`) in `settings.jsonc`.

*   **Single GPU Training:**
    ```bash
    weclone-cli train-sft
    ```

*   **Multi-GPU Training:** (Requires `deepspeed` and uncommenting the line in `settings.jsonc`)
    ```bash
    uv pip install deepspeed
    deepspeed --num_gpus=number_of_gpus weclone/train/train_sft.py
    ```
*   **Simple Inference (Browser Demo):**
    ```bash
    weclone-cli webchat-demo
    ```

*   **API Inference:**
    ```bash
    weclone-cli server
    ```

*   **Testing:**
    ```bash
    weclone-cli server
    weclone-cli test-model
    ```

---

## 🖼️ Results Showcase

*   See the original README for example output screenshots.

---

## 🤖 Deploy to Chat Bots

### AstrBot

1.  Deploy AstrBot.
2.  Deploy messaging platforms (Discord, Telegram, etc.) in AstrBot.
3.  Run `weclone-cli server`.
4.  Add a service provider in AstrBot (OpenAI type), use the WeClone API address, and set "gpt-3.5-turbo" as the model.
5.  Disable default tools in AstrBot.
6.  Set the system prompt.

### LangBot

1.  Deploy LangBot.
2.  Add a bot (Discord, Telegram, etc.) in LangBot.
3.  Run `weclone-cli server`.
4.  Add a model in LangBot (OpenAI provider),  and enter the WeClone API address.
5.  Choose the model in your pipeline.

---

## 📌 Roadmap

*   [ ] Support more data sources
*   [ ] Richer context: including contextual conversations, chat participant information, time, etc.
*   [ ] Memory support
*   [ ] Multimodal support: image support already implemented
*   [ ] Data augmentation
*   [ ] GUI support
*   [ ] COT (Chain of Thought) thinking support

---

## Troubleshooting
*   Check the [Official Documentation FAQ](https://docs.weclone.love/docs/introduce/FAQ.html).
*   Utilize [DeepWiki](https://deepwiki.com/xming521/WeClone) for problem-solving.

---

## ❤️ Contributing

Contributions (Issues/Pull Requests) are welcome!  Follow the guidelines in the original README for development environment setup (testing, type checking, and code formatting).

---

## 🙏 Acknowledgments

*   See the original README for the list of contributors.

---

## ⚠️ Disclaimer

**This project is for learning, research, and experimental purposes only.**  The original README provides a comprehensive disclaimer, including warnings about potential risks and legal considerations. Please review the full disclaimer carefully before using this project. The disclaimer covers: use at your own risk, production environment risks, model output unreliability, data security and privacy, legal compliance, and technical support limitations.

---

## ⭐ Star History

Show your support by starring the project!  (See the embedded Star History Chart in the original README for a visual).