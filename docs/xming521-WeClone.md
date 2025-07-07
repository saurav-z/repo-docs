# WeClone: Turn Chat History into a Digital Avatar 🤖

WeClone lets you create a digital avatar from your chat history, enabling you to interact with a personalized AI that reflects your unique conversational style.  Explore how to clone yourself, powered by fine-tuned LLMs! ([Original Repo](https://github.com/xming521/WeClone))

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

## Key Features:

*   **End-to-End Solution:** From chat data extraction to avatar deployment, WeClone provides a complete pipeline.
*   **Fine-tuning with Chat History:**  Train your digital twin by using your actual conversations, including images.
*   **Platform Integration:** Connect with Telegram, WeChat, and soon, WhatsApp, to create and deploy your avatar.
*   **Privacy-Focused:** Data filtering and localized fine-tuning ensure secure and controllable data handling.

## Supported Data Sources:

WeClone currently supports data from the following platforms:

| Platform   | Text | Images | Animated Emojis/Stickers | Forward | Location | Files |
| ---------- | ---- | ------ | ----------------------- | ------- | -------- | ----- |
| WeChat     | ✅   | ✅     | ❌                      | ❌       | ❌       | ❌     |
| Telegram   | ✅   | ✅     | ⚠️ Convert to Emoji       | ✅       | ✅       | ❌     |

## Important Considerations:

*   WeClone is under rapid development, and performance is subject to change.
*   Fine-tuning effectiveness depends on model size, and the quantity and quality of your chat data.  Larger models and more data generally yield better results.
*   Windows environment support is limited; using WSL is recommended.

## Recent Updates:

*   **2024-07-10:** Telegram data source added.
*   **2024-06-05:** Support for image data fine-tuning implemented.

## Hardware Requirements and VRAM Estimates

This project uses Qwen2.5-VL-7B-Instruct model by default with LoRA method for SFT stage fine-tuning. You can also use other models and methods supported by [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main#supported-models).

| Method                          | Precision |   7B  |  14B  |  30B  |   70B  |   `x`B  |
| ------------------------------- | --------- | ----- | ----- | ----- | ------ | ------- |
| Full (`bf16` or `fp16`)         |    32     | 120GB | 240GB | 600GB | 1200GB | `18x`GB |
| Full (`pure_bf16`)              |    16     |  60GB | 120GB | 300GB |  600GB |  `8x`GB |
| Freeze/LoRA/GaLore/APOLLO/BAdam |    16     |  16GB |  32GB |  64GB |  160GB |  `2x`GB |
| QLoRA                           |     8     |  10GB |  20GB |  40GB |   80GB |   `x`GB |
| QLoRA                           |     4     |   6GB |  12GB |  24GB |   48GB | `x/2`GB |
| QLoRA                           |     2     |   4GB |   8GB |  16GB |   24GB | `x/4`GB |


## Getting Started

### 1. Environment Setup

1.  **CUDA Installation:** Install CUDA (version 12.6 or above). Skip if already installed.

2.  **Python Environment:** Use [uv](https://docs.astral.sh/uv/) for a faster Python environment.
    ```bash
    git clone https://github.com/xming521/WeClone.git && cd WeClone
    uv venv .venv --python=3.10
    source .venv/bin/activate # windows .venv\Scripts\activate
    uv pip install --group main -e .
    ```

3.  **Configuration:** Copy and rename the template file:
    ```bash
    cp examples/tg.template.jsonc settings.jsonc
    ```
    All training and inference configurations are in `settings.jsonc`.

4.  **CUDA Verification (Optional):**
    ```bash
    python -c "import torch; print('CUDA Available:', torch.cuda.is_available());"
    ```

5.  **FlashAttention (Optional):** Accelerate training and inference: `uv pip install flash-attn --no-build-isolation`.

### 2. Model Download

Download models from Hugging Face or use the following command:

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct models/Qwen2.5-VL-7B-Instruct
```

### 3. Data Preparation

1.  Export chat records from [Telegram Desktop](https://desktop.telegram.org/) (Photos for message types, JSON format).
2.  Place the exported `ChatExport_*` folders into the `./dataset/telegram` directory.

### 4. Data Preprocessing

1.  Modify `language`, `platform`, and `include_type` in `settings.jsonc` as needed.
2.  Set your Telegram user ID in `telegram_args.my_id` in `settings.jsonc`.
3.  The project uses Microsoft Presidio to filter for PII. Customize `blocked_words` in `settings.jsonc` to add more filtering.
4.  Run the preprocessing command:
    ```bash
    weclone-cli make-dataset
    ```
    For more parameters, see [Data Preprocessing](https://docs.weclone.love/docs/deploy/data_preprocessing.html#related-parameters).

### 5. Configuration and Model Fine-tuning

*   Modify `model_name_or_path`, `template`, and `lora_target` in `settings.jsonc` to use other models.
*   Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` to control VRAM usage.
*   Modify `num_train_epochs`, `lora_rank`, `lora_dropout` in `train_sft_args` based on your dataset.

#### Single GPU Training

```bash
weclone-cli train-sft
```

#### Multi-GPU Training

Uncomment the `deepspeed` line in `settings.jsonc` and run:

```bash
uv pip install deepspeed
deepspeed --num_gpus=number_of_gpus weclone/train/train_sft.py
```

### 6.  Inference and Testing

*   Test suitable temperature and top_p values for webchat-demo
*   Modify `infer_args` in `settings.jsonc` for inference.
    ```bash
    weclone-cli webchat-demo
    ```

### 7. API Inference

```bash
weclone-cli server
```

### 8. Testing

```bash
weclone-cli server
weclone-cli test-model
```

## Results Showcase

> [!TIP]
> **Share your WeClone results with us on Twitter.  More examples available on [XiaoHongShu](https://www.xiaohongshu.com/user/profile/628109730000000021029de4)**

The Qwen2.5VL 32B model, trained on ~10,000 samples, achieved a loss of ~3.6:

<details>
<summary>Screenshots</summary>
<div style="display: flex; flex-wrap: wrap; gap: 10px;">
<img src="https://github.com/user-attachments/assets/b7d81f9b-ea56-4f7e-8ee5-7f4171bdc66d" alt="alt text" style="width: 52%; min-width: 150px;">
<img src="https://github.com/user-attachments/assets/62e58de8-1a73-44fc-a948-0d2e949e44a0" alt="alt text" style="width: 52%; min-width: 150px;">
<img src="https://github.com/user-attachments/assets/6bf6d0cc-7ff1-4748-a096-3850d924f954" alt="alt text" style="width: 52%; min-width: 150px;">
</div>
</details>


## Deploy to Chat Bots

### AstrBot

AstrBot ([https://github.com/AstrBotDevs/AstrBot](https://github.com/AstrBotDevs/AstrBot)) is a multi-platform LLM chatbot supporting Discord, Telegram, Slack, QQ, WeChat, etc.

1.  Deploy AstrBot.
2.  Deploy messaging platforms within AstrBot.
3.  Run `weclone-cli server` to start the API.
4.  Add a new OpenAI service provider in AstrBot, set the API Base URL, model (gpt-3.5-turbo), and any API Key.
5.  Disable default tools by sending `/tool off_all`.
6.  Set the system prompt in AstrBot.
![5](https://github.com/user-attachments/assets/19de7072-076a-4cdf-8ae6-46b9b89f536a)
> [!IMPORTANT]
> Verify large model service request parameters in the api_service logs align with those used during fine-tuning.  Disable tool plugins.

### LangBot

LangBot ([https://github.com/RockChinQ/LangBot](https://github.com/RockChinQ/LangBot)) is an open-source LLM chatbot platform.

1.  [Deploy LangBot](https://github.com/RockChinQ/LangBot/blob/master/README_EN.md#-getting-started).
2.  Add a robot in LangBot.
3.  Add a new model in the model page (gpt-3.5-turbo), select OpenAI, and set the request URL to WeClone's API address (see [documentation](https://docs.langbot.app/en/workshop/network-details.html)) and any API Key.

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/fc167dea-7c93-4d94-9c5f-db709d0320ba" />

4.  Select your newly added model in the pipeline configuration or modify the prompt.

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/dbb0fd0a-f760-42db-acd0-bb99c859b52e" />

## Roadmap

*   \[ ] Support more data sources
*   \[ ] Richer context (conversational history, participants, time).
*   \[ ] Memory support
*   \[ ] Multimodal support: image support already implemented
*   \[ ] Data augmentation
*   \[ ] GUI support
*   \[ ] COT (Chain of Thought) support

## Troubleshooting
#### [Official Documentation FAQ](https://docs.weclone.love/docs/introduce/FAQ.html)
For problem solving, it is also recommended to use [DeepWiki](https://deepwiki.com/xming521/WeClone)

## Contributing

Contributions are welcome!  Check the Issues or help review PRs. For new features, discuss them first via Issues.

Development environment:
```bash
uv pip install --group dev -e .
pre-commit install
```

Use `pytest tests` for testing, `pyright` for type checking, and `ruff` for code formatting.

## Acknowledgments

Thanks to the code contributors and community members:

<a href="https://github.com/xming521/WeClone/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xming521/WeClone" />
</a>

This project leverages open source projects like [PyWxDump](https://github.com/xaoyaoo/PyWxDump), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [AstrBot](https://github.com/AstrBotDevs/AstrBot), [LangBot](https://github.com/RockChinQ/LangBot), and others.

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