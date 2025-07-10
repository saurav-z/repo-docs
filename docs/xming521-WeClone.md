# WeClone: Create Your Digital Avatar From Chat History

**Turn your chat history into a digital twin!** WeClone is a one-stop solution for creating personalized digital avatars, allowing you to fine-tune large language models with your chat data and deploy them for various applications.  Explore the original repository at [https://github.com/xming521/WeClone](https://github.com/xming521/WeClone).

## Key Features

*   ✨ **End-to-End Solution:** From data export to deployment, WeClone provides a complete workflow for digital avatar creation.
*   💬 **Fine-tuning with Chat History:** Imbue your avatar with your unique "flavor" by fine-tuning LLMs using your chat data, including image modal data.
*   🔗 **Platform Integration:** Seamlessly integrate with Telegram, and (coming soon) WeChat and WhatsApp, to create a digital avatar from your preferred chat platforms.
*   🛡️ **Privacy-Focused:** Includes privacy information filtering and options for localized fine-tuning and deployment for secure and controllable data processing.

## Core Functionality

*   **Data Source Support:**
    | Platform | Text | Images | Animated Emojis/Stickers | Quote | Forward | Location | Files |
    | :------- | :--- | :----- | :----------------------- | :---- | :------ | :------- | :---- |
    | WeChat   | ✅    | ✅     | ❌                       | ❌     | ❌       | ❌        | ❌    |
    | Telegram | ✅    | ✅     | ⚠️ (Convert to Emoji)   | ❌     | ✅       | ✅        | ❌    |

## Getting Started

### Prerequisites
*   CUDA (version 12.6 or above)
*   Python 3.10+
*   Recommended: [uv](https://docs.astral.sh/uv/) for dependency management

### Environment Setup

1.  **Install Dependencies:**
    ```bash
    git clone https://github.com/xming521/WeClone.git && cd WeClone
    uv venv .venv --python=3.10
    source .venv/bin/activate # windows .venv\Scripts\activate
    uv pip install --group main -e .
    ```
2.  **Configure:** Copy `examples/tg.template.jsonc` to `settings.jsonc` and modify for your needs.
3.  **Verify CUDA (Optional):**
    ```bash
    python -c "import torch; print('CUDA Available:', torch.cuda.is_available());"
    ```
4.  **(Optional) Accelerate:** `uv pip install flash-attn --no-build-isolation`

### Model and Data Preparation

1.  **Download a Model:**  Recommended to use [Hugging Face](https://huggingface.co/docs/hub/models-downloading).

    ```bash
    git lfs install
    git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct models/Qwen2.5-VL-7B-Instruct
    ```

2.  **Prepare Chat Data:** Export your chat history (Telegram via Desktop, WeChat is also supported) and place them in `./dataset/telegram` directory.
3.  **Data Preprocessing:** Modify configurations in `settings.jsonc` and run:
    ```bash
    weclone-cli make-dataset
    ```

### Fine-tuning and Inference
1.  **Configure Parameters:** Adjust `model_name_or_path`, `template`, `lora_target` in `settings.jsonc`.
2.  **Single GPU Training:**
    ```bash
    weclone-cli train-sft
    ```
3.  **Multi-GPU Training:**  Uncomment `deepspeed` line and run:
    ```bash
    uv pip install deepspeed
    deepspeed --num_gpus=number_of_gpus weclone/train/train_sft.py
    ```

4.  **Simple Inference with Browser Demo:**
    ```bash
    weclone-cli webchat-demo
    ```

5.  **API Inference:**
    ```bash
    weclone-cli server
    ```

6.  **Test with Common Chat Questions:**
    ```bash
    weclone-cli server
    weclone-cli test-model
    ```

## Results Showcase

[Include some results screenshots here]
**Using the Qwen2.5VL 32B model with approximately 10,000 processed effective data samples, the loss was reduced to around 3.6:**
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

*   Follow steps to deploy [AstrBot](https://github.com/AstrBotDevs/AstrBot)
*   Follow steps to deploy messaging platforms like Discord, Telegram, Slack in AstrBot
*   Execute `weclone-cli server` to start the API service
*   Add a new service provider in AstrBot, select OpenAI type, fill in the API Base URL according to AstrBot's deployment method (e.g., for docker deployment it might be http://172.17.0.1:8005/v1), fill in the model as gpt-3.5-turbo, and enter any API Key
*   Tool calling is not supported after fine-tuning, please turn off the default tools first by sending the command: `/tool off_all` on the messaging platform, otherwise the fine-tuned effect won't be visible.
*   Set the system prompt in AstrBot according to the default_system used during fine-tuning.
![5](https://github.com/user-attachments/assets/19de7072-076a-4cdf-8ae6-46b9b89f536a)

>   [IMPORTANT]
>   Check the api_service logs to ensure that the large model service request parameters are consistent with those used during fine-tuning as much as possible, and turn off all tool plugin capabilities.

### LangBot

*   Follow steps to deploy [LangBot](https://github.com/RockChinQ/LangBot/blob/master/README_EN.md#-getting-started)
*   Add a bot (Discord, Telegram, Slack, Lark e.g.) in LangBot
*   Execute `weclone-cli server` to start the WeClone API service
*   Add a new model in the model page, name it `gpt-3.5-turbo`, select OpenAI as the provider, fill in the request URL as WeClone's address. For detailed connection methods, refer to the [documentation](https://docs.langbot.app/en/workshop/network-details.html), and enter any API Key.

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/835853ab-6ddc-459e-ae21-b04c38a85b5b" />

*   Select the model you just added in the pipeline configuration, or modify the prompt configuration

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/da61342d-84f9-4f02-87bc-3d4c7cdf187c" />

## Roadmap

*   [ ] Support more data sources
*   [ ] Richer context: including contextual conversations, chat participant information, time, etc.
*   [ ] Memory support
*   [ ] Multimodal support: image support already implemented
*   [ ] Data augmentation
*   [ ] GUI support
*   [ ] COT (Chain of Thought) thinking support

## Troubleshooting

*   [Official Documentation FAQ](https://docs.weclone.love/docs/introduce/FAQ.html)
*   Consider using [DeepWiki](https://deepwiki.com/xming521/WeClone) for problem solving.

## Contributing

Contributions are welcome!  See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

### Development environment:
```bash
uv pip install --group dev -e .
pre-commit install
```

### Testing
The project uses `pytest` for testing, `pyright` for type checking, and `ruff` for code formatting.   
Before submitting your code, you should run `pytest tests` to ensure all tests pass.


## Acknowledgments

Thanks to all code contributors and community members.
<a href="https://github.com/xming521/WeClone/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xming521/WeClone" />
</a>

Special thanks to [PyWxDump](https://github.com/xaoyaoo/PyWxDump), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [AstrBot](https://github.com/AstrBotDevs/AstrBot), [LangBot](https://github.com/RockChinQ/LangBot) and other open source projects.

## Disclaimer
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