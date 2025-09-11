<div align="center">
  <a href="https://github.com/NEKOparapa/AiNiee">
    <img src="https://github.com/NEKOparapa/AiNiee-chatgpt/blob/main/Example%20image/logo.png" width=60%>
  </a>
</div>

<div align="center">
  <a href="README_EN.md">English</a> | 简体中文
</div>

---

# AiNiee: Effortlessly Translate Games, Books, and Documents with AI

**AiNiee is your all-in-one AI translation tool, revolutionizing how you translate complex text like games, ebooks, subtitles, and documents.**

## Key Features

*   🎮 **Game Translation Powerhouse:** Deeply supports a wide array of game text extraction tools (Mtool, Renpy, Translator++, ParaTranzr, VNText, SExtractor, etc.).
*   📚 **Broad Format Support:** Handles I18Next data, Epub/TXT ebooks, Srt/Vtt/Lrc subtitles, Word/PDF/MD documents, and more.
*   🚀 **Intelligent & Efficient:** One-click operation with automatic file and language detection; get translations in minutes.
*   ⏱️ **Fast Translation:** Get your translations done in a matter of minutes.
*   🎯 **Optimized for Long Texts:** Utilizes lightweight translation formats, thought chain translation, AI terminology tables, and context association to guarantee the cohesion and accuracy of long text translations.
*   💎 **High-Quality Output:** Offers options for adjusting prompts (basic prompts, character introductions, background settings, translation styles), AI refinement, AI formatting, and terminology extraction. This meets the needs of users who demand higher translation quality.

---

## Getting Started with AiNiee: A Simple 3-Step Process

1.  **Configure Your Interface:** Set up your preferred translation API.  We offer online and offline options with detailed setup instructions.
    *   **Online Interfaces:**
        *   **DeepSeek:** Cost-effective, no GPU required, supports all languages. [DeepSeek Interface Setup](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartDeepSeek)
        *   **Huo:** Alternative to DeepSeek if DeepSeek is unavailable. [Huo Interface Setup](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartHuo)

    > <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第一步.png">

2.  **Drag & Drop Your Files:** Simply drag the folder containing your source files into the designated area.  Supports direct translation for novels, subtitles, and documents. For games, integration with text extraction tools is required.
    > <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第二步.png">

3.  **Start Translating:** Click the "Start" button and let AiNiee do the rest!
    > <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第三步.png">

    *   Download AiNiee: [https://github.com/NEKOparapa/AiNiee/releases](https://github.com/NEKOparapa/AiNiee/releases)

---

<details>
<summary>

## Game Translation Deep Dive

</summary>

<details>
<summary>

### Tool Preparation

</summary>

*   **`📖 Game Text Extraction Tools`**

    | Tool Name      | Description                                   | Project Type |
    | :------------- | :-------------------------------------------- | :----------- |
    | [Mtool](https://afdian.com/p/d42dd1e234aa11eba42452540025c377) | Easy to use, recommended for beginners.        | Mtool Export Files       |
    | [Translator++](https://dreamsavior.net/download/) | More complex, powerful, recommended for advanced users. | T++ Export or Trans Project Files |
    | [ParaTranzr](https://paratranz.cn/projects) | Moderately complex, powerful, recommended for advanced users. | ParaTranzr Export Files      |
    | [RenPy SDK](https://www.renpy.org/latest.html) | Moderately complex, powerful, recommended for advanced users. | RenPy Export Files       |

*   **`🧰 Local Model Running Tools`**

    | Tool Name                  | Description                                   |
    | :------------------------- | :-------------------------------------------- |
    | [Sakura\_Launcher\_GUI](https://github.com/PiDanShouRouZhouXD/Sakura_Launcher_GUI) | Dedicated GUI launcher for Sakura models.   |
    | [LM Studio](https://lmstudio.ai/download) | A platform for local deployment of Large Language Models (LLMs), designed to simplify LLM usage and management.   |
    | [ollama](https://ollama.com/) | Open-source cross-platform large model tool    |

</details>

<details>
<summary>

### Translation Tutorials

</summary>

*   **`📺 Game Translation Video Tutorials`**

    | Video Link                                                                                        | Description                          |
    | :----------------------------------------------------------------------------------------------- | :----------------------------------- |
    | [Mtool Tutorial](https://www.bilibili.com/video/BV1h6421c7MA)                                        | Recommended for first-time users.  |
    | [Translator++ Tutorial](https://www.bilibili.com/video/BV1LgfoYzEaX/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for first-time users.  |
    | [Wolf Game Tutorial](https://www.bilibili.com/video/BV1SnXbYiEjQ/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for first-time users.  |
    | [Name Reading Tutorial](https://www.bilibili.com/video/BV1j1VyzqERD/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for advanced users. |

*   **`🎫 Game Translation Written Tutorials`**

    | Tutorial Link                                                                                                | Description                                                                                                                               |
    | :----------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
    | [Mtool Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Mtool)       | Suitable for beginners, to simplify the translation of RPG, RenPY, Krkr games, using external translation methods.                      |
    | [Translator++ Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Translator--%EF%BC%88%E5%B7%A5%E7%A8%8B%E6%96%87%E4%BB%B6%E7%89%88%EF%BC%89) | Suitable for translating RPG, RenPY, Krkr games, and other games, for embedded translations.                                           |
    | [Paratranz Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Paratranz)   | Suitable for translating MODs of various large games.                                                                                   |
    | [StevExtraction Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90StevExtraction) | Suitable for translating RPGmakerMZ/MZ games.                                                                                           |
    | [Unity Translation Tutorial](https://zhuanlan.zhihu.com/p/1894065679927313655)                              | Suitable for translating Unity games.                                                                                                 |
    | [Comprehensive Game Translation Tutorial](https://www.notion.so/AI-1d43d31f89b280f6bd61e12580652ce5?pvs=4)  | Suitable for translating various types of games, to create high-quality embedded patches.                                            |

</details>
</details>

---

<details>
<summary>

## Feature Descriptions

</summary>

<details>
<summary>

### Settings Explained

</summary>

*   [Features - Interface Management](https://github.com/NEKOparapa/AiNiee/wiki/%E5%8A%9F%E8%83%BD%E2%80%90%E6%8E%A5%E5%8F%A3%E7%AE%A1%E7%90%86)

</details>

<details>
<summary>

### Tables Described

</summary>

*   [Table - AI Terminology Table](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E6%9C%AF%E8%AF%AD%E8%A1%A8%E4%BB%8B%E7%BB%8D)
*   [Table - AI Forbidden Translation Table](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E7%A6%81%E7%BF%BB%E8%A1%A8%E4%BB%8B%E7%BB%8D)
*   [Table - Text Replacement](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90%E6%96%87%E6%9C%AC%E6%9B%BF%E6%8D%A2%E4%BB%8B%E7%BB%8D)

</details>

<details>
<summary>

### Plugin Information

</summary>

*   [Plugin - Language Filter](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90LanguageFilter)
*   [Plugin - Text Normalizer](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90TextNormalizer)

</details>

<details>
<summary>

### Additional Notes

</summary>

*   **`Multi-key Polling`**:  Use multiple API keys to distribute the workload and speed up translation. When entering, separate each key with a comma (no line breaks). For example: `key1,key2,key3`
*   **`Batch File Translation`**:  Place all files you want to translate in the input folder; nested folder structures are also supported.
*   **`Configuration Migration`**:  Configuration information is stored in `resource/config.json`. Copy this file to the new version's `resource` folder when upgrading.

</details>
</details>

---

<details>
<summary>

## Contributing

</summary>

*   **`Develop Enhanced Plugins`**: Create powerful plugins by following the [Plugin Development Guide](https://github.com/NEKOparapa/AiNiee/blob/main/PluginScripts/README.md).

*   **`Improve or Add Support for Files`**: Requires some code programming skills; pull the source code and make improvements. File reading code is in the `ModuleFolders\FileReader` and `FileOutputer` folders.  Refer to the [Writer System Guide](https://github.com/NEKOparapa/AiNiee/blob/main/ModuleFolders/FileAccessor/README.md). UI support is in `UserInterface\Setting\ProjectSettingsPage`.

*   **`Improve Regular Expression Library`**: A complete regular expression library will greatly help with in-game embedding and future translation work. The regex library is in the [Resource\Regex](https://github.com/NEKOparapa/AiNiee/blob/main/Resource/Regex/regex.json) folder.

*   **`Improve UI Translation`**:  If the UI text in multiple languages is not accurate, please submit your suggestions or directly modify the text. Localized text is in the [Resource\Localization](https://github.com/NEKOparapa/AiNiee/tree/main/Resource/Localization) folder.

</details>

---

<details>
<summary>

## Special Statement

</summary>

AiNiee has evolved over time thanks to the continuous individual development, user feedback, and contributions of the developers. It represents a collaborative effort that has led to a mature and comprehensive AI translation system.

This AI translation tool is intended for personal and lawful use only. Any direct or indirect illegal profit-making activities using this tool are not authorized and are not supported.

*   **`Community (QQ)`**:  QQ Group (Mainly Active, Answer: github): 821624890, backup TG Group: https://t.me/+JVHbDSGo8SI2Njhl,

---

## Sponsorship

[![xxxx](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/徽章.png)](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/赞赏码.png)