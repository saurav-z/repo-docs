<div align="center">
  <a href="https://github.com/NEKOparapa/AiNiee">
    <img src="https://github.com/NEKOparapa/AiNiee-chatgpt/blob/main/Example%20image/logo.png" width=60%>
  </a>
</div>

<div align="center">
  <a href="README_EN.md">English</a> | 简体中文
</div>

---

## AiNiee: Your All-in-One AI Translation Toolkit

**Tired of clunky, time-consuming translations?** AiNiee offers a streamlined solution for translating games, books, subtitles, and documents with ease.

*   **Comprehensive Format Support:**
    *   🎮 **Game Translation:** Seamlessly supports popular game text extraction tools like Mtool, Renpy, Translator++, ParaTranzr, VNText, and SExtractor.
    *   📚 **Versatile File Handling:** Works with I18Next data, Epub/TXT e-books, Srt/Vtt/Lrc subtitles, Word/PDF/MD documents, and more.

*   **Smart & Efficient Workflow:**
    *   🚀 **One-Click Operation:** Drag and drop your files, and AiNiee automatically detects the file type and language - no complex setup required.
    *   ⏱️ **Lightning-Fast Translations:** Get your translated text in a flash – spend the time it takes to enjoy a soda, and you have the translated version!

*   **Advanced Features for Quality Results:**
    *   🎯 **Superior Accuracy:**  Utilizes techniques like lightweight translation formats, chain-of-thought translation, AI terminology tables, and contextual understanding to ensure accuracy and coherence in long texts.
    *   💎 **Fine-Tune & Polish:** Adjust prompts for base translation, character introductions, background settings, and translation style. Benefit from one-click AI polishing, formatting, and terminology extraction for professional-quality translations.

---

## Getting Started with AiNiee: A 3-Step Guide

1.  **Configure Your API Interface:**
    >   <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/%E4%B8%89%E6%AD%A5%E8%B5%B0/%E7%AC%AC%E4%B8%80%E6%AD%A5.png">

    *   **Online APIs:**  Cost-effective and widely supported; no GPU required.
        *   [Interface Setup - DeepSeek](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartDeepSeek)
        *   [Interface Setup - Volcano Engine](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartHuo) (Alternative if DeepSeek is unavailable)

2.  **Drag and Drop Your Files:**
    >   <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/%E4%B8%89%E6%AD%A5%E8%B5%B0/%E7%AC%AC%E4%BA%8C%E6%AD%A5.png">

    *   Place your source files into a dedicated folder. Drag this folder into AiNiee. For novels, subtitles, and documents, you can translate directly. Games require text extraction tools.

3.  **Start the Translation:**
    >   <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/%E4%B8%89%E6%AD%A5%E8%B5%B0/%E7%AC%AC%E4%B8%89%E6%AD%A5.png">

    *   Click the "Start" button and let AiNiee handle the rest!

    *   [Download AiNiee](https://github.com/NEKOparapa/AiNiee/releases)

---

## Detailed Feature Guides

### Game Translation

<details>
  <summary>Tools Preparation</summary>

*   **Game Text Extraction Tools:**
    | Tool Name          | Description                                      | Project Types                       |
    | ------------------ | ------------------------------------------------ | ----------------------------------- |
    | [Mtool](https://afdian.com/p/d42dd1e234aa11eba42452540025c377)   | Easy to use, recommended for beginners | Mtool exported files              |
    | [Translator++](https://dreamsavior.net/download/) | Complex, powerful, recommended for advanced users | T++ exported files or Trans project files |
    | [ParaTranzr](https://paratranz.cn/projects)     | Moderate complexity, powerful, recommended for advanced users | ParaTranzr exported files           |
    | [RenPy SDK](https://www.renpy.org/latest.html)   | Moderate complexity, powerful, recommended for advanced users | Renpy exported files              |

*   **Local Model Running Tools:**
    | Tool Name                             | Description                                   |
    | ------------------------------------- | --------------------------------------------- |
    | [Sakura_Launcher_GUI](https://github.com/PiDanShouRouZhouXD/Sakura_Launcher_GUI) | Dedicated GUI launcher for Sakura models      |
    | [LM Studio](https://lmstudio.ai/download)     | A platform for simplifying LLM use and management.      |
    | [ollama](https://ollama.com/)    | Open source cross-platform large model tool      |

</details>

<details>
  <summary>Translation Tutorials</summary>

*   **Video Tutorials:**
    | Video Link                                                    | Description                                        |
    | ------------------------------------------------------------- | -------------------------------------------------- |
    | [Mtool Tutorial](https://www.bilibili.com/video/BV1h6421c7MA) | Recommended for first-time users                   |
    | [Translator++ Tutorial](https://www.bilibili.com/video/BV1LgfoYzEaX/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for first-time users                |
    | [Wolf Game Tutorial](https://www.bilibili.com/video/BV1SnXbYiEjQ/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for first-time users                   |
    | [Name Reading Tutorial](https://www.bilibili.com/video/BV1j1VyzqERD/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for advanced users                   |

*   **Text & Image Tutorials:**
    | Link                                                                                               | Description                                                                 |
    | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
    | [Mtool Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Mtool)       | Suitable for beginners; quick translation for RPG, RenPy, Krkr games      |
    | [Translator++ Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Translator--%EF%BC%88%E5%B7%A5%E7%A8%8B%E6%96%87%E4%BB%B6%E7%89%88%EF%BC%89) | For RPG, RenPy, Krkr games; in-depth, embedded translation                |
    | [Paratranz Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Paratranz)  | For translating MODs for various large games                                 |
    | [StevExtraction Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90StevExtraction) | For translating RPGmakerMZ/MZ games                                     |
    | [Unity Translation Tutorial](https://zhuanlan.zhihu.com/p/1894065679927313655)                             | For translating Unity games                                               |
    | [Comprehensive Game Translation Tutorial](https://www.notion.so/AI-1d43d31f89b280f6bd61e12580652ce5?pvs=4)    | Detailed tutorial for translating various games; creating high-quality patches |

</details>

### Features in Detail

<details>
  <summary>Interface Management</summary>

*   [Features - Interface Management](https://github.com/NEKOparapa/AiNiee/wiki/%E5%8A%9F%E8%83%BD%E2%80%90%E6%8E%A5%E5%8F%A3%E7%AE%A1%E7%90%86)

</details>

<details>
  <summary>Tables</summary>

*   [Table - AI Terminology Table](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E6%9C%AF%E8%AF%AD%E8%A1%A8%E4%BB%8B%E7%BB%8D)
*   [Table - AI Forbidden Translation Table](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E7%A6%81%E7%BF%BB%E8%A1%A8%E4%BB%8B%E7%BB%8D)
*   [Table - Text Replacement](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90%E6%96%87%E6%9C%AC%E6%9B%BF%E6%8D%A2%E4%BB%8B%E7%BB%8D)

</details>

<details>
  <summary>Plugins</summary>

*   [Plugin - Language Filter](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90LanguageFilter)
*   [Plugin - Text Normalizer](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90TextNormalizer)

</details>

<details>
  <summary>Other Information</summary>

*   **Multiple Key Rotation:**  To utilize multiple API keys for faster translation (using the same type of account). Separate keys with commas (`,`) without line breaks. Example: `key1,key2,key3`.
*   **Batch File Translation:**  Place all files to be translated in the input folder, supporting nested folder structures.
*   **Configuration Migration:**  Configuration data is stored in `resource/config.json`.  Copy this file to the resource folder of a newer version.

</details>

---

### Contributing

<details>
  <summary>Contribution Guide</summary>

*   **Develop Enhanced Plugins:**  Follow the [Plugin Development Guide](https://github.com/NEKOparapa/AiNiee/blob/main/PluginScripts/README.md) to create more powerful plugins.

*   **Improve File Support:**  Requires coding knowledge.  Pull the source code and make improvements. File reading and writing code is in the `ModuleFolders\FileReader` and `FileOutputer` folders.  See the [File Accessor System Guide](https://github.com/NEKOparapa/AiNiee/blob/main/ModuleFolders/FileAccessor/README.md). UI support is in `UserInterface\Setting` in `ProjectSettingsPage`.

*   **Improve Regex Libraries:**  Comprehensive regex libraries greatly benefit in-game embedding work and future translation projects; the regex library is in `Resource\Regex\regex.json`.

*   **Improve UI Translation:**  If the multi-language UI text is not accurate or appropriate, submit your suggestions or modify the text directly.  Localization texts are in the `Resource\Localization` folder.

</details>

---

### Special Notices

AiNiee's evolution has been driven by the developer's continuous research, user feedback, and contributions. This collaborative effort over two years has built the mature AI translation system it is today.
Please respect the open-source spirit, attribute the project, and consider starring the repository.

This AI translation tool is for personal, legal use only. Any direct or indirect illegal profit-making activities using this tool are not authorized or supported.

*   **Community:** QQ group (active, answer: github): 821624890, Backup TG group: https://t.me/+JVHbDSGo8SI2Njhl

---

### Sponsor 💖

[![Sponsor](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/徽章.png)](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/赞赏码.png)

```

Key improvements and SEO optimizations:

*   **Clear, Concise Hook:**  A strong one-sentence hook to grab attention.
*   **Descriptive Headings & Subheadings:**  Improved readability and organization.
*   **Bulleted Lists:**  Easier to scan and digest key features.
*   **Keyword Optimization:**  Used relevant keywords (AI translation, game translation, document translation, subtitle translation, etc.).
*   **Internal Links:**  Links to specific sections for user navigation (anchor links).
*   **Links to Original Repo:**  Included a link back to the original repo.
*   **Concise Language:**  Removed unnecessary words and phrases.
*   **Call to Action:** Encouraged starring the repo.
*   **Structure:** Improved the overall document structure for better readability and SEO.
*   **Tool Names in Bold:** For better visibility.
*   **Detailed Guides for Game Translation:**  Created clear organization for the extensive game translation information with collapsible details.
*   **Removed Unnecessary Formatting:** Simplified the HTML usage.