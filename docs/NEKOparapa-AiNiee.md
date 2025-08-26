<div align="center">
  <a href="https://github.com/NEKOparapa/AiNiee">
    <img src="https://github.com/NEKOparapa/AiNiee-chatgpt/blob/main/Example%20image/logo.png" width=60%>
  </a>
</div>

<div align="center">
  <a href="README_EN.md">English</a> | 简体中文
</div>

---

## 🚀 AiNiee: Your All-in-One AI Translation Solution

**AiNiee is a powerful AI-powered translation tool designed to effortlessly translate complex text from games, books, subtitles, and documents.**  This README provides a comprehensive overview of AiNiee's features, usage, and contribution guidelines. Visit the [original repository](https://github.com/NEKOparapa/AiNiee) for the latest updates and to contribute.

---

## Key Features 🔑

*   🎮 **Game Translation Mastery:** Deeply supports game text extraction tools like Mtool, Renpy, Translator++, ParaTranzr, VNText, and SExtractor.
*   📚 **Broad Format Compatibility:** Handles a wide variety of formats including I18Next data, Epub/TXT e-books, Srt/Vtt/Lrc subtitles, and Word/PDF/MD documents.
*   ⏱️ **Fast & Efficient:** Translate large volumes of text quickly.  Get translations done in minutes!
*   🎯 **Enhanced Accuracy & Context:**  Employs advanced techniques like lightweight translation formats, chain-of-thought translation, AI terminology tables, and context association for superior translation quality.
*   💎 **Customization & Refinement:**  Supports prompt customization (e.g., basic prompts, character introductions, background settings, translation styles) and features like one-click AI refinement, formatting, and terminology extraction for high-quality results.

---

## Getting Started with AiNiee: A 3-Step Guide 🚶

1.  **Configure Interface:**
    *   **Online Interfaces:** (Paid, cost-effective, no GPU needed) Provides full language support.  See [DeepSeek Interface Setup Guide](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartDeepSeek) for details.  If DeepSeek is unavailable, try the [HuoShan Engine Interface Setup Guide](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartHuo).

    >   <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第一步.png">

2.  **Drag & Drop Files:**
    >   <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第二步.png">

    *   Place the source files in a dedicated folder and drag the folder into the designated area.  Translate novels, subtitles, and documents directly.  For games, use a text extraction tool.

3.  **Start Translation:**
    >   <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第三步.png">

    *   Click the "Start" button and let AiNiee handle the rest.
    *   [Download AiNiee](https://github.com/NEKOparapa/AiNiee/releases)

---

<details>
<summary>

## 🎮 Game Translation: Detailed Guides & Resources

</summary>

<details>
<summary>

### 🛠️ Tool Preparation

</summary>

*   **`📖 Game Text Extraction Tools`**

    | Tool Name         | Description                                               | Project Type              |
    | :---------------- | :-------------------------------------------------------- | :----------------------- |
    | [Mtool](https://afdian.com/p/d42dd1e234aa11eba42452540025c377) | Easy to use; recommended for beginners. | Mtool Exported Files      |
    | [Translator++](https://dreamsavior.net/download/) | Complex; powerful; recommended for advanced users. | T++ Export/Trans Project files |
    | [ParaTranzr](https://paratranz.cn/projects)   | Moderate complexity; powerful; recommended for advanced users. | ParaTranzr Exported Files |
    | [RenPy SDK](https://www.renpy.org/latest.html)  | Moderate complexity; powerful; recommended for advanced users. | Renpy Exported Files      |

*   **`🧰 Local Model Execution Tools`**

    | Tool Name                       | Description                                                              |
    | :------------------------------- | :----------------------------------------------------------------------- |
    | [Sakura\_Launcher\_GUI](https://github.com/PiDanShouRouZhouXD/Sakura_Launcher_GUI) | Dedicated GUI launcher for Sakura models.                           |
    | [LM Studio](https://lmstudio.ai/download)  | A platform for deploying Large Language Models (LLMs) locally. |
    | [ollama](https://ollama.com/) | Open source cross-platform large model tool |

</details>

<details>
<summary>

### 🎓 Translation Tutorials

</summary>

*   **`📺 Video Tutorials`**

    | Video Link                                                                                                | Description                                       |
    | :-------------------------------------------------------------------------------------------------------- | :------------------------------------------------ |
    | [Mtool Tutorial](https://www.bilibili.com/video/BV1h6421c7MA)                                             | Recommended for first-time users.                |
    | [Translator++ Tutorial](https://www.bilibili.com/video/BV1LgfoYzEaX/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for first-time users.                |
    | [Wolf Game Tutorial](https://www.bilibili.com/video/BV1SnXbYiEjQ/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501)    | Recommended for first-time users.                |
    | [Name Reading Tutorial](https://www.bilibili.com/video/BV1j1VyzqERD/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for advanced translation.            |

*   **`🎫 Text & Image Tutorials`**

    | Link                                                                                                           | Description                                                                      |
    | :------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- |
    | [Mtool Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Mtool)   | Suitable for beginners translating RPG, RenPY, Krkr games.          |
    | [Translator++ Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Translator--%EF%BC%88%E5%B7%A5%E7%A8%8B%E6%96%87%E4%BB%B6%E7%89%88%EF%BC%89) | Suitable for translating RPG, RenPY, Krkr games using embedded translation. |
    | [Paratranz Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Paratranz)    | Suitable for translating MODs of various large games.                               |
    | [StevExtraction Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90StevExtraction) | Suitable for translating RPGmakerMZ/MZ games.                                   |
    | [Unity Translation Tutorial](https://zhuanlan.zhihu.com/p/1894065679927313655)                                      | Suitable for translating Unity games.                                         |
    | [Comprehensive Game Translation Tutorial](https://www.notion.so/AI-1d43d31f89b280f6bd61e12580652ce5?pvs=4)        | Suitable for various games, creating high-quality embedded patches.                     |

</details>
</details>

---

<details>
<summary>

## ⚙️ Functionality Overview

</summary>

<details>
<summary>

### ⚙️ Settings Explained

</summary>

*   [Function - Interface Management](https://github.com/NEKOparapa/AiNiee/wiki/%E5%8A%9F%E8%83%BD%E2%80%90%E6%8E%A5%E5%8F%A3%E7%AE%A1%E7%90%86)

</details>

<details>
<summary>

### 📊 Table Descriptions

</summary>

*   [Table - AI Terminology Table](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E6%9C%AF%E8%AF%AD%E8%A1%A8%E4%BB%8B%E7%BB%8D)
*   [Table - AI Forbidden Words Table](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E7%A6%81%E7%BF%BB%E8%A1%A8%E4%BB%8B%E7%BB%8D)
*   [Table - Text Replacement](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90%E6%96%87%E6%9C%AC%E6%9B%BF%E6%8D%A2%E4%BB%8B%E7%BB%8D)

</details>

<details>
<summary>

### ➕ Plugin Details

</summary>

*   [Plugin - Language Filter](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90LanguageFilter)
*   [Plugin - Text Normalizer](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90TextNormalizer)

</details>

<details>
<summary>

### ℹ️ Additional Information

</summary>

*   ` Multiple Key Rotation`
    > To distribute workload and speed up translation using multiple keys, use keys from the same account type and separate them with commas, without line breaks. For example: key1,key2,key3

*   ` Batch File Translation`
    >  Place all the files you want to translate in the input folder, and it supports multi-folder structures.

*   ` Configuration Migration`
    >  Configuration information is stored in resource/config.json; copy it to the resource folder of the new version.

</details>
</details>

---

<details>
<summary>

## 🤝 Contributing

</summary>

*   **`Develop Enhanced Plugins`**: Create feature-rich plugins following the [Plugin Writing Guide](https://github.com/NEKOparapa/AiNiee/blob/main/PluginScripts/README.md).

*   **`Improve or Add File Support`**:  Requires code skills to modify the source code. The file reading code is located in the ModuleFolders\\FileReader and FileOutputer folders. See the [Writer System Guide](https://github.com/NEKOparapa/AiNiee/blob/main/ModuleFolders/FileAccessor/README.md). UI support is in UserInterface\Setting's ProjectSettingsPage.

*   **`Enhance Regex Library`**: Improve the Regular Expression Library, which significantly aids game embedding and facilitates future translation tasks. The regex library is located in the [Resource\Regex](https://github.com/NEKOparapa/AiNiee/blob/main/Resource/Regex/regex.json) folder.

*   **`Improve Interface Translations`**: Improve UI text, and contribute your suggestions or directly make edits. Localized text is in the [Resource\Localization](https://github.com/NEKOparapa/AiNiee/tree/main/Resource/Localization) folder.

</details>

---

<details>
<summary>

## 📢 Special Notice

</summary>

AiNiee's development and evolution have been driven by individual contributions, user feedback, and the collaborative efforts of the community.  This project is a testament to continuous improvement.

This AI translation tool is for *personal, lawful use only*. Any activities involving direct or indirect illegal profit generation using this tool are *not* authorized and will not be supported.

*   **`Community`**: Join our QQ group (answer: github) at 821624890, or the backup TG group: https://t.me/+JVHbDSGo8SI2Njhl

</details>

---

## 🙏 Sponsorship
[![xxx](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/徽章.png)](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/赞赏码.png)