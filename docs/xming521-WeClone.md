# WeClone: Create Your Digital Avatar from Chat History

**Transform your chat history into a personalized digital avatar!** [Explore WeClone on GitHub](https://github.com/xming521/WeClone).

## Key Features

*   💬 **Chat History to Avatar:** A complete end-to-end solution, from chat data export to model deployment.
*   💫 **Fine-tuned LLMs:**  Fine-tune Large Language Models (LLMs) using your chat data, infusing your avatar with your unique "flavor" (image modal support).
*   🔗 **Multi-Platform Support:** Integrate with Telegram, WeChat, and WhatsApp (coming soon) to create your digital avatar.
*   🛡️ **Privacy Focused:** Offers privacy information filtering with localized fine-tuning and deployment for secure and controllable data.

## Core Capabilities & Highlights
*   **Data Source Platforms:** Supports WeChat and Telegram.
    | Platform | Text | Images | Forward | Animated Emojis/Stickers |
    |----------|------|--------|---------|--------------------------|
    | WeChat | ✅ | ✅ | ❌ | ❌ |
    | Telegram | ✅ | ✅ | ✅ | ⚠️ Convert to Emoji |

## Get Started

### Environment Setup

1.  **CUDA Installation:** Install CUDA (version 12.6 or above), if not already present.
2.  **Python Environment:** Use [uv](https://docs.astral.sh/uv/) for efficient dependency management:
    ```bash
    git clone https://github.com/xming521/WeClone.git && cd WeClone
    uv venv .venv --python=3.10
    source .venv/bin/activate # windows .venv\Scripts\activate
    uv pip install --group main -e .
    ```
3.  **Configuration:** Copy the configuration template and rename it to `settings.jsonc`:
    ```bash
    cp examples/tg.template.jsonc settings.jsonc
    ```
4.  **CUDA Verification:** (Optional for Mac) Test CUDA configuration:
    ```bash
    python -c "import torch; print('CUDA Available:', torch.cuda.is_available());"
    ```
5.  **(Optional) FlashAttention:** Install for accelerated training and inference: `uv pip install flash-attn --no-build-isolation`.

### Model Download
Download models through Hugging Face, or by running:
```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct models/Qwen2.5-VL-7B-Instruct
```

### Data Preparation

1.  **Export Chat Records:** Export chat history from Telegram Desktop as photos and JSON.
2.  **Organize Data:** Place exported `ChatExport_*` folders into the `./dataset/telegram` directory.

### Data Preprocessing

1.  **Configure:** Modify the `language`, `platform`, and `include_type` within the `settings.jsonc` file to suit your needs.
2.  **Telegram ID:** If using Telegram, update `telegram_args.my_id` in `settings.jsonc` with your Telegram user ID.
3.  **Privacy Protection:**  Utilizes Microsoft Presidio to automatically remove potentially sensitive data. You can customize the `blocked_words` in `settings.jsonc` to manually filter specific words or phrases.
4.  **Process Data:** Execute the following command to process your data:
    ```bash
    weclone-cli make-dataset
    ```

### Fine-tuning and Inference

*   Modify `model_name_or_path`, `template`, `lora_target` in `settings.jsonc` to select the model to be fine-tuned.
*   Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` for VRAM usage.
*   Fine-tune the model with the command `weclone-cli train-sft` (single GPU) or using `deepspeed` (multi-GPU):
    ```bash
    uv pip install deepspeed
    deepspeed --num_gpus=number_of_gpus weclone/train/train_sft.py
    ```
*   Test model using `weclone-cli webchat-demo`.
*   Use API with the command `weclone-cli server`.
*   Test with questions via the command `weclone-cli test-model`.

## Results Showcase

See the Qwen2.5VL 32B model results with screenshots.

## Deployment Options

### AstrBot
An easy-to-use multi-platform LLM chatbot and development framework ✨ Supports Discord, Telegram, Slack, QQ, WeChat, Enterprise WeChat, Feishu and other platforms.
1.  Deploy AstrBot
2.  Deploy messaging platforms like Discord, Telegram, Slack in AstrBot
3.  Execute `weclone-cli server` to start the API service
4.  Add a new service provider in AstrBot, select OpenAI type, fill in the API Base URL according to AstrBot's deployment method (e.g., for docker deployment it might be http://172.17.0.1:8005/v1), fill in the model as gpt-3.5-turbo, and enter any API Key
5.  Tool calling is not supported after fine-tuning, please turn off the default tools first by sending the command: `/tool off_all` on the messaging platform, otherwise the fine-tuned effect won't be visible.
6.  Set the system prompt in AstrBot according to the default_system used during fine-tuning.

### LangBot
A platform that enables users to quickly set up an IM bot
1. [Deploy LangBot](https://github.com/RockChinQ/LangBot/blob/master/README_EN.md#-getting-started)
2. Add a bot (Discord, Telegram, Slack, Lark e.g.) in LangBot
3. Execute `weclone-cli server` to start the WeClone API service
4. Add a new model in the model page, name it `gpt-3.5-turbo`, select OpenAI as the provider, fill in the request URL as WeClone's address. For detailed connection methods, refer to the [documentation](https://docs.langbot.app/en/workshop/network-details.html), and enter any API Key.
5. Select the model you just added in the pipeline configuration, or modify the prompt configuration

## Roadmap

*   [ ] Support more data sources
*   [ ] Richer context: including contextual conversations, chat participant information, time, etc.
*   [ ] Memory support
*   [ ] Multimodal support: image support already implemented
*   [ ] Data augmentation
*   [ ] GUI support
*   [ ] COT (Chain of Thought) thinking support

## Troubleshooting

See the [Official Documentation FAQ](https://docs.weclone.love/docs/introduce/FAQ.html) or use [DeepWiki](https://deepwiki.com/xming521/WeClone).

## Contributing

Contributions are welcome!  See the [CONTRIBUTING](https://github.com/xming521/WeClone/blob/main/CONTRIBUTING.md) guide.

## Acknowledgments

Thanks to contributors and the open-source projects: [PyWxDump](https://github.com/xaoyaoo/PyWxDump), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [AstrBot](https://github.com/AstrBotDevs/AstrBot), [LangBot](https://github.com/RockChinQ/LangBot).

## ⭐ Star History
[![Star History Chart](https://api.star-history.com/svg?repos=xming521/WeClone&type=Date)](https://www.star-history.com/#xming521/WeClone&Date)

## ⚠️ Disclaimer

**This project is for research and experimental purposes only. Use at your own risk.  See the full disclaimer in the original [README](https://github.com/xming521/WeClone) for details regarding potential risks and responsibilities.**