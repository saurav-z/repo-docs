<div align="center">
  <a href="https://github.com/NEKOparapa/AiNiee">
    <img src="https://github.com/NEKOparapa/AiNiee-chatgpt/blob/main/Example%20image/logo.png" width=60%>
  </a>
</div>

<div align="center">
  <a href="README_EN.md">English</a> | 简体中文
</div>

---

## 🚀 AiNiee: The AI Translation Tool for Everyone

**Tired of tedious manual translation?** AiNiee is your all-in-one solution for effortlessly translating games, books, subtitles, and documents with AI!

*   **🎮 Extensive Format Support:** Translate a wide variety of file types, including game texts from Mtool, Renpy, Translator++, ParaTranzr, VNText, SExtractor, and more.
*   **📚 Versatile Compatibility:** Process I18Next data, Epub/TXT ebooks, Srt/Vtt/Lrc subtitles, Word/PDF/MD documents, and more with ease.
*   **⏱️ Intelligent & Efficient:** One-click operation with automatic language detection saves you time. Get translations in minutes!
*   **🎯 Optimized for Long Texts:** Leverage advanced features like lightweight translation formats, thought chain translation, AI terminology, and contextual association for superior accuracy in complex texts.
*   **💎 Enhanced Quality Control:** Fine-tune your translations with features like prompt adjustments for base prompts, character introductions, background settings, and translation styles. Includes AI polishing, layout, and terminology extraction for professional-quality results.

---

## 📝 Getting Started with AiNiee: Three Simple Steps

1.  **Configure Your API:**  (Choose your preferred AI translation provider)

    *   [接口设置说明 - DeepSeek](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartDeepSeek) (Paid, high cost-effectiveness, all languages supported)
    *   [接口设置说明 - 火山引擎](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartHuo) (Alternative if DeepSeek is unavailable)

    > <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第一步.png">

2.  **Drag and Drop Files:** Place the files you want to translate into a dedicated folder and drag that folder into the AiNiee interface.  (Works directly for books, subtitles, and documents; requires text extraction tools for games.)

    > <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第二步.png">

3.  **Start Translating:** Click the "Start" button and let AiNiee handle the rest.

    > <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第三步.png">

    *   [AiNiee Download](https://github.com/NEKOparapa/AiNiee/releases)

---

## 🕹️ Game Translation Guide

<details>
<summary>
  
### 🎮 Game Translation Details
</summary>

This section provides detailed guidance and resources for translating games using AiNiee.

<details>
<summary>

### ⚙️ Tool Preparation
</summary>

Essential tools for extracting game text and running local models.

 *   **`📖 Game Text Extraction Tools`**

    | Tool Name        | Description                                     | Project Type       |
    | :-------------- | :---------------------------------------------- | :----------------- |
    | [Mtool](https://afdian.com/p/d42dd1e234aa11eba42452540025c377)      | Easy to use, recommended for beginners             | Mtool Export Files |
    | [Translator++](https://dreamsavior.net/download/) | Complex, powerful, recommended for advanced users | T++ or Trans Project Files |
    | [ParaTranzr](https://paratranz.cn/projects)      | Moderate complexity, powerful, recommended for advanced users | ParaTranzr Export Files |
    | [RenPy SDK](https://www.renpy.org/latest.html) | Moderate complexity, powerful, recommended for advanced users | Renpy Export Files |

 *   **`🧰 Local Model Running Tools`**

    | Tool Name                               | Description                                                                   |
    | :-------------------------------------- | :---------------------------------------------------------------------------- |
    | [Sakura_Launcher_GUI](https://github.com/PiDanShouRouZhouXD/Sakura_Launcher_GUI) | Dedicated GUI launcher for Sakura models                                        |
    | [LM Studio](https://lmstudio.ai/download)  | Platform to simplify LLM use and management                                              |
    | [ollama](https://ollama.com/)      | Open-source cross-platform large model tool                            |

</details>

<details>
<summary>

### 🎬 Translation Tutorials
</summary>

Video and text tutorials to guide you through the game translation process.

 *   **`📺 Video Tutorials`**

    | Video Link                                                                            | Description                                     |
    | :------------------------------------------------------------------------------------ | :---------------------------------------------- |
    | [Mtool Tutorial](https://www.bilibili.com/video/BV1h6421c7MA)                          | Recommended for first-time users                 |
    | [Translator++ Tutorial](https://www.bilibili.com/video/BV1LgfoYzEaX/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for first-time users                 |
    | [Wolf Game Tutorial](https://www.bilibili.com/video/BV1SnXbYiEjQ/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for first-time users                 |
    | [Name Reading Tutorial](https://www.bilibili.com/video/BV1j1VyzqERD/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for advanced users                 |

 *   **`🎫 Text and Image Tutorials`**

    | Video Link                                                                     | Description                                                                                                          |
    | :----------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- |
    | [Mtool Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Mtool) | Suitable for beginners, lazy translation of RPG, RenPY, Krkr and other games.                                  |
    | [Translator++ Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Translator--%EF%BC%88%E5%B7%A5%E7%A8%8B%E6%96%87%E4%BB%B6%E7%89%88%EF%BC%89) | Suitable for translating RPG, RenPY, Krkr and other games, for embedded translation.                                  |
    | [Paratranz Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Paratranz) | Suitable for translating MODs of various large games.                                                                     |
    | [StevExtraction Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90StevExtraction) | Suitable for translating RPGmakerMZ/MZ games.                                                                     |
    | [Unity Translation Tutorial](https://zhuanlan.zhihu.com/p/1894065679927313655) | Suitable for translating Unity games.                                                                            |
    | [Comprehensive Game Translation Tutorial](https://www.notion.so/AI-1d43d31f89b280f6bd61e12580652ce5?pvs=4) | Suitable for translating various games and producing high-quality embedded patches.                                 |

</details>

</details>

---

## ⚙️ Features Explained

<details>
<summary>

### ⚙️ Feature Details
</summary>

*   **Interfaces:** Explore various API configuration options.
    *   [Function ‐ Interface Management](https://github.com/NEKOparapa/AiNiee/wiki/%E5%8A%9F%E8%83%BD%E2%80%90%E6%8E%A5%E5%8F%A3%E7%AE%A1%E7%90%86)
  

*   **Tables:** Learn more about AiNiee's various tables.
    *   [Table - AI Terminology Table](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E6%9C%AF%E8%AF%AD%E8%A1%A8%E4%BB%8B%E7%BB%8D)
    *   [Table - AI Forbidden Translation Table](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E7%A6%81%E7%BF%BB%E8%A1%A8%E4%BB%8B%E7%BB%8D)
    *   [Table - Text Replacement](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90%E6%96%87%E6%9C%AC%E6%9B%BF%E6%8D%A2%E4%BB%8B%E7%BB%8D)

*   **Plugins:**  Enhance your translation workflow with powerful plugins.
    *   [Plugin - Language Filter](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90LanguageFilter)
    *   [Plugin - Text Normalizer](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90TextNormalizer)

*   **Other Notes:**
    *   ` Multi-key Round Robin ` Use multiple keys to share the load.  Use keys of the same type and separate them with commas, without line breaks (e.g., key1,key2,key3).
    *   ` Batch File Translation ` Place all files to be translated in the input folder, supporting multi-folder structures.
    *   ` Configuration Migration ` Configuration information is stored in resource/config.json.  Copy it to the new version's resource folder when upgrading.
</details>

---

## ✨ Contributing

<details>
<summary>

### ✨ Contribution Guidelines
</summary>

Help improve AiNiee!

*   **`Enhance Plugins`**: Develop more powerful plugins by following the [Plugin Writing Guide](https://github.com/NEKOparapa/AiNiee/blob/main/PluginScripts/README.md).
*   **`Improve File Support`**: Contribute to the project by improving existing file reader and writer modules and adding new file formats.  Requires coding skills. The file reading/writing code is in the ModuleFolders\FileReader and FileOutputer folders.  See [File System Writing Guide](https://github.com/NEKOparapa/AiNiee/blob/main/ModuleFolders/FileAccessor/README.md). UI support is in UserInterface\Setting's ProjectSettingsPage.
*   **`Improve Regular Expression Library`**:  A complete regular expression library helps with game embedding. Regular expressions are in [Resource\Regex](https://github.com/NEKOparapa/AiNiee/blob/main/Resource/Regex/regex.json).
*   **`Improve UI Translations`**: Suggest improvements or directly edit the UI text translations.  Localization files are in [Resource\Localization](https://github.com/NEKOparapa/AiNiee/tree/main/Resource/Localization).

</details>

---

## ⚠️ Important Disclaimer

AiNiee is the result of continuous development, user feedback, and community contributions. Please respect the open-source spirit.  Give credit to the project and star the repository!

This AI translation tool is for personal, legal use only. Any direct or indirect illegal profit-making activities using this tool are not authorized and will not be supported or recognized.

*   **`Community`**: Join our QQ group (active, answer: github): 8216248九零, and TG group: https://t.me/+JVHbDSGo8SI2Njhl

---

## 🙏 Sponsorship

[![Sponsor Button](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/徽章.png)](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/赞赏码.png)

[Back to Top](#top)
```
Key improvements and explanations:

*   **SEO Optimization:**  Keywords like "AI translation," "game translation," "subtitle translation," and related terms are incorporated naturally throughout the text, especially in the headings and summaries. The title is more descriptive.
*   **Clear Structure:**  Uses headings, subheadings, and bullet points for readability and easy scanning.  This makes it easier for users to quickly understand the features and how to use the tool.
*   **Concise Summaries:** The core functionality is explained in a clear, one-sentence hook, with each feature broken down.
*   **Action-Oriented Language:** Phrases like "Get translations in minutes!", "One-click operation," and "Effortlessly translate" are more engaging.
*   **Call to Action (Download Link):** The download link is prominently featured.
*   **Detailed Game Translation Section:** A separate section is created for game translation, the primary use case, which includes links to tools, videos, and tutorials.
*   **Contribution & Disclaimer Sections:** Explains how users can help and the legal limitations.
*   **Sponsorship:**  The call for sponsorship is still present.
*   **Link to the Original Repo:**  The link to the original repo is in the first line.
*   **Clearer Instructions:** The "Getting Started" section simplifies the instructions.
*   **Markdown Formatting:** The use of markdown makes the README visually appealing and easy to read.
*   **Removed redundant text:** Removed redundant text to make the text more clean.
*   **Contextualization:**  Provides the user with the context of each part.