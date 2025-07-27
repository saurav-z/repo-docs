# WeClone: Create Your Digital Avatar from Chat History

**Effortlessly transform your chat history into a personalized digital avatar, ready to interact like you, powered by AI.**  [Explore the project on GitHub](https://github.com/xming521/WeClone)

![download](https://github.com/user-attachments/assets/cd4a87c6-1649-4ce5-bce8-bd5b08b278de)

## Key Features

*   **End-to-End Solution:**  A complete pipeline for digital avatar creation, encompassing data export, preprocessing, model training, and deployment.
*   **LLM Fine-tuning:**  Fine-tune large language models (LLMs) using your chat history, infusing your avatar with your unique communication style.
*   **Platform Integration:**  Seamlessly integrate with Telegram and Discord (more platforms coming soon!) to bring your digital avatar to life.
*   **Privacy Focused:**  Data privacy is prioritized with features like information filtering and localized deployment for secure and controlled data handling.

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

## Platform Support

**Data Source Support:**

| Platform   | Text | Images |  Forward | Location | Files | Animated Emojis/Stickers | Links (Sharing) | Quote | Voice | Video |
| :-------- | :--: | :----: | :----: | :----: | :---: | :---------------------: | :-------------: | :---: | :---: | :---: |
| Telegram   |  ✅  |   ✅   |   ✅   |   ✅   |   ❌   |      ⚠️ Convert to Emoji       |       ❌       |  ❌  |  ❌  |  ❌  |
| WhatsApp   |  🚧  |   🚧   |   🚧   |   🚧   |   🚧   |           🚧          |       🚧       |  🚧  |  🚧  |  🚧  |
| Discord    |  🚧  |   🚧   |   🚧   |   🚧   |   🚧   |           🚧          |       🚧       |  🚧  |  🚧  |  🚧  |
| Slack      |  🚧  |   🚧   |   🚧   |   🚧   |   🚧   |           🚧          |       🚧       |  🚧  |  🚧  |  🚧  |

**Deployment Platform Support:**

| Platform   | Deployment Support |
| :-------- | :----------------: |
| Telegram   |        ✅        |
| WhatsApp   |        🚧        |
| Discord    |        ✅        |
| Slack      |        ✅        |

> [!IMPORTANT]
> *   WeClone is under active development; performance may not reflect final results.
> *   LLM fine-tuning effectiveness depends on model size and data quality.  Larger models and more data generally yield better results.

### Recent Updates
[25/07/10] Data source added Telegram   
[25/06/05] Support for image modal data fine-tuning

## Hardware Requirements

The project defaults to the Qwen2.5-VL-7B-Instruct model with the LoRA method for SFT fine-tuning. You can use other models and methods supported by [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main#supported-models).

Estimated VRAM requirements:
| Method                          | Precision |   7B  |  14B  |  30B  |   70B  |   `x`B  |
| ------------------------------- | --------- | ----- | ----- | ----- | ------ | ------- |
| Full (`bf16` or `fp16`)         |    32     | 120GB | 240GB | 600GB | 1200GB | `18x`GB |
| Full (`pure_bf16`)              |    16     |  60GB | 120GB | 300GB |  600GB |  `8x`GB |
| Freeze/LoRA/GaLore/APOLLO/BAdam |    16     |  16GB |  32GB |  64GB |  160GB |  `2x`GB |
| QLoRA                           |     8     |  10GB |  20GB |  40GB |   80GB |   `x`GB |
| QLoRA                           |     4     |   6GB |  12GB |  24GB |   48GB | `x/2`GB |
| QLoRA                           |     2     |   4GB |   8GB |  16GB |   24GB | `x/4`GB |

## Getting Started

### Environment Setup

1.  **CUDA Installation:** (skip if already installed, **requires version 12.6 or above**)

2.  **Python Environment:**
    It is recommended to use [uv](https://docs.astral.sh/uv/) to install dependencies, which is a very fast Python environment manager. After installing uv, you can use the following commands to create a new Python environment and install dependencies.

    ```bash
    git clone https://github.com/xming521/WeClone.git && cd WeClone
    uv venv .venv --python=3.10
    source .venv/bin/activate # windows .venv\Scripts\activate
    uv pip install --group main -e .
    ```

3.  **Configuration:**
    Copy the template configuration file and rename it to `settings.jsonc`, then customize the settings within this file.

    ```bash
    cp examples/tg.template.jsonc settings.jsonc
    ```

    > [!NOTE]
    > Training and inference configurations are unified in the file `settings.jsonc`

4.  **CUDA Verification (optional):**
    Verify CUDA setup with:

    ```bash
    python -c "import torch; print('CUDA Available:', torch.cuda.is_available());"
    ```

5.  **(Optional) FlashAttention:**
    Install FlashAttention for faster training and inference: `uv pip install flash-attn --no-build-isolation`.

### Model Download

Use [Hugging Face](https://huggingface.co/docs/hub/models-downloading) to download models, or run:

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct models/Qwen2.5-VL-7B-Instruct
```

### Data Preparation

1.  **Export Chat Records:** Use [Telegram Desktop](https://desktop.telegram.org/) to export chat history. Select "Photos" for message types and "JSON" for format.  Place the exported `ChatExport_*` folders into `./dataset/telegram`.

### Data Preprocessing

1.  **Configure:** Modify `language`, `platform`, and `include_type` in `settings.jsonc`.
2.  **Telegram Specific:** Set `telegram_args.my_id` in `settings.jsonc` to your Telegram user ID.
3.  **Privacy Filtering:**  The project uses Microsoft Presidio to remove sensitive information.  A `blocked_words` list in `settings.jsonc` allows you to add custom filter terms.

    > [!IMPORTANT]
    > 🚨 Please be sure to protect personal privacy and do not leak personal information!

4.  **Run Preprocessing:**

    ```bash
    weclone-cli make-dataset
    ```

    More Parameter Details: [Data Preprocessing](https://docs.weclone.love/docs/deploy/data_preprocessing.html#related-parameters)

### Model Fine-tuning

1.  **(Optional) Model and Parameters:** Adjust `model_name_or_path`, `template`, `lora_target` in `settings.jsonc`.  Modify `per_device_train_batch_size`, `gradient_accumulation_steps` to manage VRAM.  Customize `num_train_epochs`, `lora_rank`, `lora_dropout` based on your data.

#### Single GPU Training
```bash
weclone-cli train-sft
```

#### Multi-GPU Training
Uncomment the `deepspeed` line in `settings.jsonc` and use the following command for multi-GPU training:
```bash
uv pip install "deepspeed<=0.16.9"
deepspeed --num_gpus=number_of_gpus weclone/train/train_sft.py
```

### Simple Inference

Test temperature and top\_p values, then modify `infer_args` in settings.jsonc.
```bash
weclone-cli webchat-demo
```

### Inference using API

```bash
weclone-cli server
```

### Testing

Test common chat questions, results are in test_result-my.txt.
```bash
weclone-cli server
weclone-cli test-model
```

## 🖼️ Results Showcase
> [!TIP]
> **We're looking for interesting examples of native English speakers chatting with WeClone! Feel free to share them with us on Twitter.**

## 🤖 Deploy to Chat Bots

### AstrBot
[AstrBot](https://github.com/AstrBotDevs/AstrBot) is an easy-to-use multi-platform LLM chatbot and development framework ✨ Supports Discord, Telegram, Slack, Feishu and other platforms.

Usage steps:
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

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/de44e6e3-3a53-44d9-af76-96364cfca30f" />

1.  [Deploy LangBot](https://github.com/RockChinQ/LangBot/blob/master/README_EN.md#-getting-started)
2.  Add a bot (Discord, Telegram, Slack, Lark e.g.) in LangBot
3.  Execute `weclone-cli server` to start the WeClone API service
4.  Add a new model in the model page, name it `gpt-3.5-turbo`, select OpenAI as the provider, fill in the request URL as WeClone's address. For detailed connection methods, refer to the [documentation](https://docs.langbot.app/en/workshop/network-details.html), and enter any API Key.

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/835853ab-6ddc-459e-ae21-b04c38a85b5b" />

6.  Select the model you just added in the pipeline configuration, or modify the prompt configuration

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/da61342d-84f9-4f02-87bc-3d4c7cdf187c" />

## 📌 Roadmap

*   \[ ] Support more data sources
*   \[ ] Richer context: including contextual conversations, chat participant information, time, etc.
*   \[ ] Memory support
*   \[ ] Multimodal support: image support already implemented
*   \[ ] Data augmentation
*   \[ ] GUI support
*   \[ ] COT (Chain of Thought) thinking support

## Troubleshooting
#### [Official Documentation FAQ](https://docs.weclone.love/docs/introduce/FAQ.html)
It is also recommended to use [DeepWiki](https://deepwiki.com/xming521/WeClone) for problem solving.

## ❤️ Contributing

Contributions are welcome!  See the Issues or help review Pull Requests.  For new feature additions, please discuss them via Issues first.
Development environment:
```bash
uv pip install --group dev -e .
pre-commit install
```
The project uses `pytest` for testing, `pyright` for type checking, and `ruff` for code formatting.
Before submitting your code, you should run `pytest tests` to ensure all tests pass.

## 🙏 Acknowledgments

This project leverages the following open source projects: [PyWxDump](https://github.com/xaoyaoo/PyWxDump), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [AstrBot](https://github.com/AstrBotDevs/AstrBot), [LangBot](https://github.com/RockChinQ/LangBot), and many others.  Thanks also to all code contributors and community members!

<a href="https://github.com/xming521/WeClone/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xming521/WeClone" />
</a>

## ⚠️ Disclaimer
> [!CAUTION]
> **This project is for learning, research and experimental purposes only. There are significant risks in using it for production environments, please assess carefully. Do not use for illegal purposes, consequences are at your own risk.**

> [!IMPORTANT]
> #### WeClone is currently not partnered with any platform and has not issued any cryptocurrency. The only official website is: [weclone.love](https://www.weclone.love). Beware of imitations.

<details>
<summary>Click to view disclaimer terms</summary>

### 1. Use at Your Own Risk
- Users should fully understand and bear all related risks when using this project
- **The project authors are not responsible for any direct or indirect losses arising from the use of this project**
- Including but not limited to: data loss, financial loss, legal disputes, personal reputation damage, social relationship impact, psychological trauma, career development obstacles, business reputation damage, etc.

### 2. Production Environment Risk Warning
- **Use for commercial purposes or providing external services requires bearing all risks yourself**
- All consequences that may result from production environment use (including but not limited to service interruption, data security issues, user complaints, legal liability, etc.) are entirely borne by the user
- **It is recommended to conduct thorough testing, verification and risk assessment before using in production environments**

### 3. Model Output Unreliability
- Fine-tuned models may produce inaccurate, harmful or misleading content
- Model outputs do not represent the views or intentions of real persons
- Users should conduct manual review and verification of model outputs

### 4. Data Security and Privacy
- Users should ensure that uploaded chat records and other data comply with relevant laws and regulations
- Users should obtain **appropriate authorization from data-related persons**
- This project is not responsible for **data leakage or privacy infringement**

### 5. Legal Compliance
- **Users should ensure that using this project complies with local laws and regulations**
- Involving artificial intelligence, data protection, intellectual property and other related laws
- **Users bear the consequences of illegal use**

### 6. Technical Support Limitations
- This project is provided "as is" without any express or implied warranties
- Authors do not promise to provide continuous technical support or maintenance
- No guarantee of project stability, reliability or applicability

## Usage Recommendations

### Mandatory Bot Identity Identification
**When using digital avatars generated by this project, it is strongly recommended to:**
- Clearly identify as "AI Bot" or "Digital Avatar" at the beginning of each conversation
- Prominently mark "AI-generated content" in the user interface
- Avoid letting users mistake it for real human conversation, which could cause risks

### Risk Assessment Recommendations

If you must use in production environments, it is recommended to:
1. Conduct comprehensive security testing
2. Establish complete content review mechanisms
3. Develop emergency response plans
4. Purchase appropriate insurance coverage
5. Consult legal professionals for advice


This disclaimer may be revised with project updates, users should regularly check the latest version. Continuing to use this project indicates agreement with the latest disclaimer terms.

**Once you download, clone, modify, distribute or use the code or models of this project in any way, it indicates that you have fully read, understood and agreed to unconditionally accept all terms of this disclaimer.**

</details>

**Please carefully read and understand all contents of this disclaimer, ensuring strict compliance with relevant regulations when using this project.**
<br>

## ⭐ Star History
> [!TIP]
> If this project is helpful to you, or if you are interested in the future development of this project, please give the project a Star, thank you

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=xming521/WeClone&type=Date)](https://www.star-history.com/#xming521/WeClone&Date)

</div>