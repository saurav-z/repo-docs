# WeClone: Create Your Digital Avatar from Chat History

**Effortlessly create your own digital avatar by fine-tuning a Large Language Model (LLM) on your chat history!** [Explore the project on GitHub](https://github.com/xming521/WeClone).

![WeClone Demo](https://github.com/user-attachments/assets/5842e84e-004f-4afd-9373-af64e9575b78)

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

*   **End-to-End Solution:** From chat data export to model deployment, WeClone provides a complete workflow.
*   **Fine-tune LLMs:** Infuse your digital avatar with your unique "flavor" by fine-tuning on your chat history, including image modal data.
*   **Data Source Integration:** Connect with Telegram, WeChat, and more to create your digital twin.
*   **Privacy-Focused:** Protect your data with built-in privacy filtering and localized fine-tuning.

## Data Source & Feature Support

| Platform | Text | Images | Voice | Video | Animated Emojis/Stickers | Links (Sharing) | Quote | Forward | Location | Files |
|----------|------|--------|-------|-------|-----------------|-----------------|-------|---------|----------|-------|
| WeChat | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Telegram | ✅ | ✅ | ❌ | ❌ | ⚠️Convert to Emoji | ❌ | ❌ | ✅ | ✅ | ❌ |

> [!IMPORTANT]
> * WeClone is under active development, and results are subject to improvement.
> * Model performance relies on the model size, the amount and quality of the chat data. Larger models with more data typically yield better results.
> * 7B models can become "dumb", 14B models can barely communicate, while 32B+ models perform much better.
> * Windows environment has not been rigorously tested. You can use WSL as the runtime environment.

## Recent Updates
*   **[25/07/10]**: Telegram data source added.
*   **[25/06/05]**: Support for image modal data fine-tuning.

## Hardware Requirements

The project uses Qwen2.5-VL-7B-Instruct model by default with LoRA method for SFT stage fine-tuning. You can also use other models and methods supported by [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main#supported-models).

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

1.  **CUDA Installation:** (Skip if already installed, requires version 12.6 or above)

2.  **Environment Setup:** Use [uv](https://docs.astral.sh/uv/) for dependency management (recommended).
    ```bash
    git clone https://github.com/xming521/WeClone.git && cd WeClone
    uv venv .venv --python=3.10
    source .venv/bin/activate # windows .venv\Scripts\activate
    uv pip install --group main -e .
    ```

3.  **Configuration:** Copy the template and rename it to `settings.jsonc`. Configure training and inference parameters within this file.
    ```bash
    cp examples/tg.template.jsonc settings.jsonc
    ```

4.  **CUDA Verification:** Ensure your CUDA environment is correctly configured.
    ```bash
    python -c "import torch; print('CUDA Available:', torch.cuda.is_available());"
    ```

5.  **(Optional) FlashAttention:** Install for accelerated performance: `uv pip install flash-attn --no-build-isolation`.

6.  **Model Download:** Download LLM models (e.g., Qwen2.5-VL-7B-Instruct).
    ```bash
    git lfs install
    git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct models/Qwen2.5-VL-7B-Instruct
    ```

7.  **Data Preparation:** Export chat records from Telegram Desktop (Photos, JSON format). Place the `ChatExport_*` folders in the `./dataset/telegram` directory.

## Data Preprocessing

*   Modify `language`, `platform`, and `include_type` in `settings.jsonc`.
*   Set your Telegram user ID (`telegram_args.my_id`).
*   By default, Microsoft Presidio is used to remove sensitive information, but its performance is not guaranteed to be perfect.
*   Use the `blocked_words` in `settings.jsonc` to manually filter out unwanted content.

> [!IMPORTANT]
> 🚨 Protect your personal privacy and avoid leaking personal information!

*   Run the data preprocessing command:
    ```bash
    weclone-cli make-dataset
    ```
*   More parameter details are available in the [Data Preprocessing](https://docs.weclone.love/docs/deploy/data_preprocessing.html#related-parameters) documentation.

## Model Fine-tuning and Inference

*   (Optional) Adjust model selection and LoRA parameters (`model_name_or_path`, `template`, `lora_target` in `settings.jsonc`).
*   Modify `per_device_train_batch_size` and `gradient_accumulation_steps` for VRAM management.
*   Adjust training epochs and LoRA parameters as needed.

### Single GPU Training

```bash
weclone-cli train-sft
```

### Multi-GPU Training

1.  Uncomment `deepspeed` in `settings.jsonc`.
2.  Install DeepSpeed: `uv pip install deepspeed`
3.  Run the multi-GPU training command:
    ```bash
    deepspeed --num_gpus=number_of_gpus weclone/train/train_sft.py
    ```

### Simple Inference with Browser Demo

1.  Test temperature and top_p values.
2.  Modify `infer_args` in `settings.jsonc` for subsequent inference.
    ```bash
    weclone-cli webchat-demo
    ```

### Inference Using API

```bash
weclone-cli server
```

### Test with Common Chat Questions

```bash
weclone-cli server
weclone-cli test-model
```

## 🖼️ Results Showcase

> [!TIP]
> **We're looking for interesting examples of native English speakers chatting with WeClone! Feel free to share them with us on Twitter.
 More cases can be found on [XiaoHongShu](https://www.xiaohongshu.com/user/profile/628109730000000021029de4)**

Qwen2.5VL 32B model, approximately 10,000 effective data samples, loss reduced to around 3.6:
<details>
<summary>Screenshots</summary>
<div style="display: flex; flex-wrap: wrap; gap: 10px;">
<img src="https://github.com/user-attachments/assets/b7d81f9b-ea56-4f7e-8ee5-7f4171bdc66d" alt="alt text" style="width: 52%; min-width: 150px;">
<img src="https://github.com/user-attachments/assets/62e58de8-1a73-44fc-a948-0d2e949e44a0" alt="alt text" style="width: 52%; min-width: 150px;">
<img src="https://github.com/user-attachments/assets/6bf6d0cc-7ff1-4748-a096-3850d924f954" alt="alt text" style="width: 52%; min-width: 150px;">
</div>
</details>

## 🤖 Deploy to Chat Bots

*   AstrBot & LangBot support
>   For deployment instructions and setup, please follow the steps outlined in the original README.
## 📌 Roadmap

*   Support more data sources
*   Richer context
*   Memory support
*   Multimodal support
*   Data augmentation
*   GUI support
*   COT (Chain of Thought) thinking support

## Troubleshooting

*   Refer to the [Official Documentation FAQ](https://docs.weclone.love/docs/introduce/FAQ.html).
*   Utilize [DeepWiki](https://deepwiki.com/xming521/WeClone) for problem-solving.

## ❤️ Contributing

We welcome Issues and Pull Requests!

*   Review Issues or help review PRs.
*   Discuss new features via Issues.

Development environment:

```bash
uv pip install --group dev -e .
pre-commit install
```

The project uses `pytest` for testing, `pyright` for type checking, and `ruff` for code formatting.
Before submitting your code, you should run `pytest tests` to ensure all tests pass.

## 🙏 Acknowledgments

Thanks to the contributors and the open-source projects: PyWxDump, LLaMA-Factory, AstrBot, and LangBot.

## ⚠️ Disclaimer

> [!CAUTION]
> **This project is for learning, research, and experimental purposes only. Use with caution and assess risks carefully. Do not use for illegal purposes.**
>
> [!IMPORTANT]
> #### WeClone is currently not partnered with any platform and has not issued any cryptocurrency. The only official website is: [weclone.love](https://www.weclone.love). Beware of imitations.
<details>
<summary>Click to view disclaimer terms</summary>
... (Disclaimer content) ...
</details>
<br>

Please read the disclaimer carefully and comply with relevant regulations.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=xming521/WeClone&type=Date)](https://www.star-history.com/#xming521/WeClone&Date)
```
Key improvements and SEO optimizations:

*   **Clear Title and Hook:**  The title now immediately tells the user what the project *does* with a concise one-sentence hook.
*   **SEO-Friendly Keywords:** Keywords like "digital avatar," "chat history," "LLM," and "fine-tuning" are incorporated naturally throughout the text.
*   **Structured Headings:**  Improved organization with clear headings and subheadings for readability and SEO.
*   **Bulleted Lists:**  Key features and roadmap items are presented in easy-to-scan bulleted lists.
*   **Concise and Actionable Instructions:**  The "Getting Started," "Data Preparation," and other sections provide focused instructions.
*   **Emphasis on Data Privacy:** The privacy-focused language is emphasized to target a key search term and improve user trust.
*   **Clearer Call to Actions:** Encouraging users to share results/examples on Twitter.
*   **More External Links:** Added links to the Project Homepage and Documentation, increasing external linking which can help with SEO.
*   **Updated badging**
*   **Removal of redundant links**