# WeClone: Create Your Digital Avatar from Chat History

**Turn your chat history into a personalized AI avatar with WeClone!** Explore the power of fine-tuning Large Language Models (LLMs) using your conversation data to create a unique digital representation of yourself.  Get started today and bring your digital self to life!  [Visit the GitHub Repository](https://github.com/xming521/WeClone)

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/xming521/WeClone?style=for-the-badge&logo=github&label=Stars&logoColor=white&color=ffda65)](https://github.com/xming521/WeClone/stargazers)
[![GitHub release](https://img.shields.io/github/v/release/xming521/WeClone?style=for-the-badge&logo=github&label=Release&logoColor=white&color=06d094)](https://github.com/xming521/WeClone/releases)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/+JEdak4m0XEQ3NGNl)
[![Twitter](https://img.shields.io/badge/Twitter-@weclone567-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/weclone567)
[![小红书](https://img.shields.io/badge/WeClone-FE2C55?style=for-the-badge&logo=xiaohongshu&logoColor=white)](https://www.xiaohongshu.com/user/profile/628109730000000021029de4)
<a href="https://qm.qq.com/cgi-bin/qm/qr?k=wNdgbOVT6oFOJ2wlMLsolUXErW9ESLpk&jump_from=webapi&authKey=z/reOp6YLyvR4Tl2k2nYMsLoMC3w9/99ucgKMX0oRGlxDV/WbYnvq2QxODoIkfxn" target="_blank" style="text-decoration: none;">
  <img src="https://img.shields.io/badge/QQ群-708067078-12B7F5?style=for-the-badge&logo=qq&logoColor=white" alt="WeClone①" title="WeClone①">
</a>


<a href="https://hellogithub.com/repository/12ab209b56cb4cfd885c8cfd4cfdd53e" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=12ab209b56cb4cfd885c8cfd4cfdd53e&claim_uid=RThlPDoGrFvdMY5" alt="Featured｜HelloGitHub" style="width: 150px; height: 28px;" /></a>
<a href="https://trendshift.io/repositories/13759" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13759" alt="xming521%2FWeClone | Trendshift" style="width: 220px; height: 50px;" /></a>
<a href="https://deepwiki.com/xming521/WeClone"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"  style="width: 134px; height: 23px;margin-bottom: 3px;"></a>
</div>

<p align="center">
  <a href="https://github.com/xming521/WeClone/blob/master/README_zh.md" target="_blank">简体中文</a>｜
  English</a>｜
  <a href="https://www.weclone.love/" target="_blank"> Project Homepage </a> ｜
  <a href="https://docs.weclone.love/docs/introduce/what-is-weclone.html" target="_blank"> Documentation </a>
</p>

> [!IMPORTANT]
> ### Telegram is now supported as a data source !

## Key Features

*   **End-to-End Solution:** From chat data export to model deployment, WeClone provides a complete pipeline for digital avatar creation.
*   **Fine-tuned LLMs:** Train LLMs on your chat history to capture your unique communication style, with image support.
*   **Multi-Platform Integration:** Seamlessly integrate with Telegram, WeChat, and more (WhatsApp coming soon) to create your avatar.
*   **Privacy-Focused:** Includes privacy information filtering and localized fine-tuning for secure and controlled data management.

## Supported Data Sources

| Platform   | Text | Images | Animated Emojis/Stickers |  Forward  |
| :---------- | :---: | :----: | :---------------------: | :-------: |
| WeChat     |  ✅  |   ✅   |            ❌           |     ❌     |
| Telegram   |  ✅  |   ✅   |   ⚠️Convert to Emoji   |     ✅     |

**Note:** WeClone is in active development; performance may not reflect final results.  The effectiveness of fine-tuning depends on model size, data quantity, and quality.

## Recent Updates

*   **2024/07/10:** Added Telegram data source support.
*   **2024/06/05:**  Implemented support for image data fine-tuning.

## Hardware Requirements

WeClone defaults to the Qwen2.5-VL-7B-Instruct model with LoRA for SFT.  You can also use other models and methods supported by [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main#supported-models).

**Estimated VRAM Requirements:** See the original README for the detailed table.

## Getting Started

1.  **CUDA Installation:** Ensure you have CUDA installed (version 12.6 or above). Skip if already installed.

2.  **Environment Setup:**
    *   Use [uv](https://docs.astral.sh/uv/) for dependency management:
        ```bash
        git clone https://github.com/xming521/WeClone.git && cd WeClone
        uv venv .venv --python=3.10
        source .venv/bin/activate  # or .venv\Scripts\activate for Windows
        uv pip install --group main -e .
        ```

3.  **Configuration:**
    *   Copy the configuration file template:
        ```bash
        cp examples/tg.template.jsonc settings.jsonc
        ```
    *   Edit `settings.jsonc` to configure training and inference.

4.  **CUDA Verification (Optional):**  Confirm CUDA is configured correctly:
    ```bash
    python -c "import torch; print('CUDA Available:', torch.cuda.is_available());"
    ```

5.  **(Optional) FlashAttention:** Install for acceleration: `uv pip install flash-attn --no-build-isolation`.

## Model Download

Use [Hugging Face](https://huggingface.co/docs/hub/models-downloading) or the following command:

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct models/Qwen2.5-VL-7B-Instruct
```

## Data Preparation

1.  **Export Chat Records:** Export chat history from [Telegram Desktop](https://desktop.telegram.org/) as JSON (Photos for message types).
2.  **Organize Data:** Place exported `ChatExport_*` folders in the `./dataset/telegram` directory.

## Data Preprocessing

*   Modify `language`, `platform`, and `include_type` in `settings.jsonc`.
*   Set your Telegram user ID in `telegram_args.my_id`.
*   WeClone uses Microsoft Presidio to remove sensitive information, but a blocklist (`blocked_words`) in `settings.jsonc` allows you to add custom filters.

> [!IMPORTANT]
> 🚨 Protect your personal privacy!

*   Run the data preprocessing command:
    ```bash
    weclone-cli make-dataset
    ```
    More Parameter Details: [Data Preprocessing](https://docs.weclone.love/docs/deploy/data_preprocessing.html#related-parameters)

## Configure Parameters and Fine-tune Model

*   (Optional) Modify `model_name_or_path`, `template`, `lora_target` in `settings.jsonc`.
*   Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` for VRAM.
*   Modify `num_train_epochs`, `lora_rank`, `lora_dropout` in `train_sft_args`.

### Single GPU Training

```bash
weclone-cli train-sft
```

### Multi-GPU Training

Uncomment the `deepspeed` line in `settings.jsonc` and run:

```bash
uv pip install deepspeed
deepspeed --num_gpus=number_of_gpus weclone/train/train_sft.py
```

### Simple Inference with Browser Demo

Test and adjust temperature and top_p, then modify `infer_args` in `settings.jsonc`.
```bash
weclone-cli webchat-demo
```

### Inference Using API

```bash
weclone-cli server
```

### Test with Common Chat Questions

Test results are in test_result-my.txt.
```bash
weclone-cli server
weclone-cli test-model
```

## 🖼️ Results Showcase

> [!TIP]
> **Share your WeClone conversations with native English speakers on Twitter! Find more examples on [XiaoHongShu](https://www.xiaohongshu.com/user/profile/628109730000000021029de4)**

[See the original README for screenshot examples.]

## 🤖 Deploy to Chat Bots

### AstrBot

[AstrBot](https://github.com/AstrBotDevs/AstrBot) is an easy-to-use multi-platform LLM chatbot and development framework ✨ Supports Discord, Telegram, Slack, QQ, WeChat, Enterprise WeChat, Feishu and other platforms.

1.  Deploy AstrBot
2.  Deploy messaging platforms like Discord, Telegram, Slack in AstrBot
3.  Execute `weclone-cli server` to start the API service
4.  Add a new service provider in AstrBot, select OpenAI type, fill in the API Base URL according to AstrBot's deployment method (e.g., for docker deployment it might be http://172.17.0.1:8005/v1), fill in the model as gpt-3.5-turbo, and enter any API Key
5.  Tool calling is not supported after fine-tuning, please turn off the default tools first by sending the command: `/tool off_all` on the messaging platform, otherwise the fine-tuned effect won't be visible.
6.  Set the system prompt in AstrBot according to the default_system used during fine-tuning.
![5](https://github.com/user-attachments/assets/19de7072-076a-4cdf-8ae6-46b9b89f536a)
> [!IMPORTANT]
> Check the api_service logs to ensure that the large model service request parameters are consistent with those used during fine-tuning as much as possible, and turn off all tool plugin capabilities.

### LangBot

[LangBot](https://github.com/langbot-app/LangBot) is an easy-to-use open-source LLM chatbot platform suitable for various scenarios. It connects to various global instant messaging platforms. You can set up your IM bot in just 5 minutes.

[See the original README for the LangBot image.]

1.  [Deploy LangBot](https://github.com/RockChinQ/LangBot/blob/master/README_EN.md#-getting-started)
2.  Add a bot (Discord, Telegram, Slack, Lark e.g.) in LangBot
3.  Execute `weclone-cli server` to start the WeClone API service
4.  Add a new model in the model page, name it `gpt-3.5-turbo`, select OpenAI as the provider, fill in the request URL as WeClone's address. For detailed connection methods, refer to the [documentation](https://docs.langbot.app/en/workshop/network-details.html), and enter any API Key.

[See the original README for the LangBot images.]

6.  Select the model you just added in the pipeline configuration, or modify the prompt configuration

## 📌 Roadmap

*   \[ ] Support more data sources
*   \[ ] Richer context: contextual conversations, chat participant information, time, etc.
*   \[ ] Memory support
*   \[ ] Multimodal support: image support already implemented
*   \[ ] Data augmentation
*   \[ ] GUI support
*   \[ ] COT (Chain of Thought) thinking support

## Troubleshooting

#### [Official Documentation FAQ](https://docs.weclone.love/docs/introduce/FAQ.html)

It is also recommended to use [DeepWiki](https://deepwiki.com/xming521/WeClone) for problem solving.

## ❤️ Contributing

Contributions are welcome!  Check the Issues or help review Pull Requests. Discuss new features via Issues first.

Development environment:
```bash
uv pip install --group dev -e .
pre-commit install
```

The project uses `pytest` for testing, `pyright` for type checking, and `ruff` for code formatting.
Before submitting code, run `pytest tests`.

## 🙏 Acknowledgments

Thanks to the following code contributors and other community members for their contributions

[See the original README for the contributor graph.]

This project also benefits from excellent open source projects such as [PyWxDump](https://github.com/xaoyaoo/PyWxDump), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [AstrBot](https://github.com/AstrBotDevs/AstrBot), [LangBot](https://github.com/RockChinQ/LangBot), and others.

## ⚠️ Disclaimer

> [!CAUTION]
> **This project is for learning, research and experimental purposes only. There are significant risks in using it for production environments, please assess carefully. Do not use for illegal purposes, consequences are at your own risk.**

> [!IMPORTANT]
> #### WeClone is currently not partnered with any platform and has not issued any cryptocurrency. The only official website is: [weclone.love](https://www.weclone.love). Beware of imitations.

[See the original README for the detailed disclaimer terms.]

## ⭐ Star History

> [!TIP]
> If this project is helpful to you, or if you are interested in the future development of this project, please give the project a Star, thank you

[See the original README for the Star History Chart.]

```

Key improvements:

*   **SEO Optimization:**  Added keywords like "digital avatar," "LLM," "chat history," "fine-tuning" in headings and throughout the text.
*   **Clear Structure:** Uses headings, bullet points, and numbered lists for readability.
*   **Concise Language:**  Rewritten text to be more direct and easier to understand.
*   **Actionable:** Provides clear "how-to" instructions.
*   **Strong Hook:**  The opening sentence immediately grabs the reader's attention.
*   **Call to Action:** Encourages users to "get started."
*   **Markdown Formatting:**  Correctly formatted for GitHub README rendering.
*   **Summarization:** Condenses the original README while preserving essential information.
*   **Emphasis on Key Features:** Highlights the most important aspects.
*   **Cleaned up layout and made it easier to read**