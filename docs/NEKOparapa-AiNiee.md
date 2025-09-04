<div align="center">
  <a href="https://github.com/NEKOparapa/AiNiee">
    <img src="https://github.com/NEKOparapa/AiNiee-chatgpt/blob/main/Example%20image/logo.png" width=60%>
  </a>
</div>

<div align="center">
  <a href="README_EN.md">English</a> | 简体中文
</div>

---

# AiNiee: Your All-in-One AI Translation Tool 🚀

**Tired of tedious translation tasks? AiNiee streamlines the process, making AI translation a breeze for games, ebooks, subtitles, and more!**  Explore the original repository on [GitHub](https://github.com/NEKOparapa/AiNiee) for more information.

---

## Key Features

*   🎮 **Game Translation Powerhouse**:
    *   Extensive support for game text extraction tools like Mtool, Renpy, and more.
*   📚 **Broad Format Compatibility**:
    *   Handles I18Next data, Epub/TXT ebooks, Srt/Vtt/Lrc subtitles, Word/PDF/MD documents, and more.
*   🚀 **Smart & Efficient**:
    *   One-click operation with automatic file and language detection.
    *   Get translations in a flash – as fast as it takes to drink a soda!
*   🎯 **Optimized for Long Texts**:
    *   Utilizes techniques like light translation formats, chain-of-thought translation, AI terminology tables, and contextual association for accurate and coherent results.
    *   Fine-tune translations with features such as AI refinement, AI formatting, and terminology extraction.

---

## AiNiee: The Three-Step Translation Process 📢

1.  **Configure API:**
    *   <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第一步.png">
    *   Choose from online APIs (paid, cost-effective, and no GPU needed) or local models.
        *   [DeepSeek API Setup Guide](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartDeepSeek)
        *   [Huo API Setup Guide (Alternative)](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartHuo)

2.  **Drag & Drop Files:**
    *   <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第二步.png">
    *   Place your source files in a folder and drag the folder into AiNiee. Novels, subtitles, and documents can be translated directly. Games require text extraction tools.

3.  **Start Translation:**
    *   <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第三步.png">
    *   Click the "Start" button and let AiNiee do the work.

    *   [Download AiNiee](https://github.com/NEKOparapa/AiNiee/releases)

---

<details>
<summary>
  
## Game Translation Guide [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#游戏翻译)
</summary>

<details>
<summary>
  
### Tools Required
</summary>

*   **`📖 Game Text Extraction Tools`**:

    | Tool Name        | Description                                                                | Project Types          |
    | :--------------- | :------------------------------------------------------------------------- | :--------------------- |
    | [Mtool](https://afdian.com/p/d42dd1e234aa11eba42452540025c377) | Easy to use, recommended for beginners.                                | Mtool export files      |
    | [Translator++](https://dreamsavior.net/download/) | Complex, powerful; recommended for advanced users.                     | T++ export files or Trans project files |
    | [ParaTranzr](https://paratranz.cn/projects)   | Moderate learning curve, powerful; recommended for advanced users.       | ParaTranzr export files |
    | [RenPy SDK](https://www.renpy.org/latest.html)  | Moderate learning curve, powerful; recommended for advanced users.       | renpy export files      |

*   **`🧰 Local Model Running Tools`**:

    | Tool Name                                           | Description                                                                    |
    | :-------------------------------------------------- | :----------------------------------------------------------------------------- |
    | [Sakura\_Launcher\_GUI](https://github.com/PiDanShouRouZhouXD/Sakura_Launcher_GUI) | Dedicated GUI launcher for Sakura models.                                      |
    | [LM Studio](https://lmstudio.ai/download)        | A platform for local LLM deployment, simplifying LLM usage and management.    |
    | [ollama](https://ollama.com/)                     | Open-source cross-platform large model tool.                                 |
</details>

<details>
<summary>
  
### Translation Tutorials
</summary>

*   **`📺 Video Tutorials`**:

    | Video Link                                                                            | Description                                         |
    | :------------------------------------------------------------------------------------ | :-------------------------------------------------- |
    | [Mtool Tutorial](https://www.bilibili.com/video/BV1h6421c7MA)                       | Recommended for initial use.                       |
    | [Translator++ Tutorial](https://www.bilibili.com/video/BV1LgfoYzEaX/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for initial use.                       |
    | [Wolf Game Tutorial](https://www.bilibili.com/video/BV1SnXbYiEjQ/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501)  | Recommended for initial use.                       |
    | [Name Reading Tutorial](https://www.bilibili.com/video/BV1j1VyzqERD/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for advanced translation.           |

*   **`🎫 Text and Image Tutorials`**:

    | Link                                                                                                                 | Description                                                                                                      |
    | :------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
    | [Mtool Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Mtool) | Suitable for beginners; use for RPG, RenPY, Krkr, and similar games using external translation methods.     |
    | [Translator++ Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Translator--%EF%BC%88%E5%B7%A5%E7%A8%8B%E6%96%87%E4%BB%B6%E7%89%88%EF%BC%89) | Suitable for translating RPG, RenPY, Krkr, etc. games using embedded translation methods.                       |
    | [Paratranz Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Paratranz) | Suitable for translating MODs of various large-scale games.                                                       |
    | [StevExtraction Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90StevExtraction) | Suitable for translating RPGmakerMZ/MZ games.                                                                     |
    | [Unity Translation Tutorial](https://zhuanlan.zhihu.com/p/1894065679927313655)                                       | Suitable for translating Unity games.                                                                            |
    | [Comprehensive Game Translation Tutorial](https://www.notion.so/AI-1d43d31f89b280f6bd61e12580652ce5?pvs=4)            | Suitable for translating various games and creating high-quality embedded patches.                                |
</details>

</details>

---

<details>
<summary>
  
## Functionality Explained [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#功能说明)
</summary>

<details>
<summary>
  
### Settings Overview
</summary>

*   [Features - Interface Management](https://github.com/NEKOparapa/AiNiee/wiki/%E5%8A%9F%E8%83%BD%E2%80%90%E6%8E%A5%E5%8F%A3%E7%AE%A1%E7%90%86)
</details>

<details>
<summary>
  
### Tables Explained
</summary>

*   [Table - AI Terminology Table](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E6%9C%AF%E8%AF%AD%E8%A1%A8%E4%BB%8B%E7%BB%8D)
*   [Table - AI Forbidden Translation Table](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E7%A6%81%E7%BF%BB%E8%A1%A8%E4%BB%8B%E7%BB%8D)
*   [Table - Text Replacement](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90%E6%96%87%E6%9C%AC%E6%9B%BF%E6%8D%A2%E4%BB%8B%E7%BB%8D)
</details>

<details>
<summary>
  
### Plugins Explained
</summary>

*   [Plugin - Language Filter](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90LanguageFilter)
*   [Plugin - Text Normalizer](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90TextNormalizer)
</details>

<details>
<summary>
  
### Other Important Information
</summary>

*   `Multiple Key Rotation`
    > To use multiple keys to share the processing load and speed up translation, use keys of the same type. Separate keys with commas without line breaks. For example: key1,key2,key3
*   `Batch File Translation`
    > Place all files to be translated in the input folder, and it supports a multi-folder structure.
*   `Configuration Migration`
    > Configuration information is stored in resource's config.json.  You can copy this file to the new version's resource folder when updating.
</details>
</details>

---

<details>
<summary>
  
## Contributing Guidelines [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#贡献指南)
</summary>

*   **`Develop Enhanced Plugins`**: Develop plugins with expanded features according to the [plugin writing guide](https://github.com/NEKOparapa/AiNiee/blob/main/PluginScripts/README.md).
*   **`Improve or Add File Support`**: Requires some code programming skills; pull the source code to improve.  Code for reading and writing files are located in the `ModuleFolders\FileReader` and `FileOutputer` folders. [Writing Guide](https://github.com/NEKOparapa/AiNiee/blob/main/ModuleFolders/FileAccessor/README.md).  UI support is in `UserInterface\Setting\ProjectSettingsPage`.
*   **`Improve Regular Expression Library`**:  A comprehensive regex library greatly aids in game embedding and benefits future game translation efforts. The regex library is in the [Resource\Regex](https://github.com/NEKOparapa/AiNiee/blob/main/Resource/Regex/regex.json) folder.
*   **`Improve UI Translation`**:  You can submit suggestions for improving the UI text in multiple languages or make direct modifications.  Localization texts are located in the [Resource\Localization](https://github.com/NEKOparapa/AiNiee/tree/main/Resource/Localization) folder.
</details>

---

<details>
<summary>
  
## Special Notes [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#特别声明)
</summary>

AiNiee has evolved through the ongoing personal development, user feedback, and community contributions. It's been a collaborative process over two years, resulting in the mature AI translation system it is today.

Please respect the open-source spirit and give the project a star.

This AI translation tool is only intended for personal, legal use. Any direct or indirect illegal profit-making activities using this tool are outside the scope of authorization and will not be supported or endorsed.

*   **`Community Group`**:  QQ group (main active group, answer: github): 821624890, Backup TG group: https://t.me/+JVHbDSGo8SI2Njhl
</details>

---

## Sponsorship💖
[![Sponsor](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/徽章.png)](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/赞赏码.png)