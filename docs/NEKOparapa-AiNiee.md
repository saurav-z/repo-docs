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

**AiNiee empowers you to effortlessly translate games, books, subtitles, and documents with advanced AI, making complex text translation a breeze.** Explore its powerful features below!

[View the original repository](https://github.com/NEKOparapa/AiNiee)

---

## ✨ Key Features

*   **Extensive Format Support:**
    *   🎮 **Game Translation:** Seamlessly supports various game text extraction tools including Mtool, Renpy, Translator++, ParaTranzr, VNText, and SExtractor.
    *   📚 **Diverse Content Compatibility:** Effortlessly handles I18Next data, Epub/TXT ebooks, Srt/Vtt/Lrc subtitles, Word/PDF/MD documents, and more.

*   **Intelligent & Efficient:**
    *   🚀 **One-Click Operation:** Simply drag and drop, and AiNiee automatically detects files and languages, eliminating the need for manual configuration.
    *   ⏱️ **Lightning-Fast Translation:** Get your translations done in a matter of minutes.

*   **Enhanced Quality for Long Texts:**
    *   🎯 **Overcoming Limitations:** Utilizes techniques like lightweight translation formats, thought chain translation, AI terminology, and context association to ensure coherent and accurate long-text translations.
    *   💎 **Quality Enhancement Tools:** Supports customizable prompt adjustments (e.g., base prompts, character introductions, background settings, translation styles), AI refinement, AI layout, and term extraction for users who demand superior translation quality.

---

## 🎬 Getting Started with AiNiee: A 3-Step Guide

1.  **Configure Your API Interface:**
    *   [Interface Setup - DeepSeek](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartDeepSeek): Online API - Recommended for its cost-effectiveness and all-language support. No GPU required.
    *   [Interface Setup - Huo](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartHuo):  Alternative API option if DeepSeek is unavailable.

    > <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/%E4%B8%89%E6%AD%A5%E8%B5%B0/%E7%AC%AC%E4%B8%80%E6%AD%A5.png">

2.  **Drag and Drop Your Files:**
    *   Place the original files in a dedicated folder and drag the folder into the designated area. Novels, subtitles, and documents can be translated directly. For games, you'll need to use a text extraction tool first.

    > <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/%E4%B8%89%E6%AD%A5%E8%B5%B0/%E7%AC%AC%E4%BA%8C%E6%AD%A5.png">

3.  **Start Translating!**
    *   Click the "Start" button and await the completion of your translation.

    > <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/%E4%B8%89%E6%AD%A5%E8%B5%B0/%E7%AC%AC%E4%B8%89%E6%AD%A5.png">
    *   [Download AiNiee](https://github.com/NEKOparapa/AiNiee/releases)

---

<details>
<summary>
  
## 🎮 Game Translation Guide
</summary>

<details>
<summary> 

### 🛠️ Required Tools
</summary>

 * **`📖 Game Text Extraction Tools`**

      | Tool | Description | Project Type |
      |:----:|:-----:|:-----:|
      |[Mtool](https://afdian.com/p/d42dd1e234aa11eba42452540025c377)| Easy to use, recommended for beginners | Mtool Export File |
      |[Translator++](https://dreamsavior.net/download/)| Complex setup, powerful features, recommended for advanced users | T++ Export File or Trans Project File |
      |[ParaTranzr](https://paratranz.cn/projects)| Moderate learning curve, powerful features, recommended for advanced users | ParaTranzr Export File |
      |[RenPy SDK](https://www.renpy.org/latest.html)| Moderate learning curve, powerful features, recommended for advanced users | Renpy Export File |

 * **`🧰 Local Model Execution Tools`**

      | Tool | Description |
      |:----:|:-----:|
      |[Sakura_Launcher_GUI](https://github.com/PiDanShouRouZhouXD/Sakura_Launcher_GUI)| GUI Launcher for Sakura models |
      |[LM Studio](https://lmstudio.ai/download) | Platform for local deployment of large language models (LLMs), simplifies LLM usage and management. |
      |[ollama](https://ollama.com/) | Open source cross-platform large model tool |

</details>

<details>
<summary>
  
### 🎬 Translation Tutorials
</summary>

 * **`📺 Video Tutorials`**

      | Video Link | Description |
      |:----:|:-----:|
      |[Mtool Tutorial](https://www.bilibili.com/video/BV1h6421c7MA) | Recommended for initial use |
      |[Translator++ Tutorial](https://www.bilibili.com/video/BV1LgfoYzEaX/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501)| Recommended for initial use |
      |[Wolf Game Tutorial](https://www.bilibili.com/video/BV1SnXbYiEjQ/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501)| Recommended for initial use |
      |[Name Reading Tutorial](https://www.bilibili.com/video/BV1j1VyzqERD/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501)| Recommended for advanced users |

 * **`🎫 Text & Image Tutorials`**

      | Tutorial Link | Description |
      |:----:|:-----:|
      |[Mtool Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Mtool) | Suitable for beginners, lazy translation of RPG, RenPY, Krkr games, using external translation |
      |[Translator++ Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Translator--%EF%BC%88%E5%B7%A5%E7%A8%8B%E6%96%87%E4%BB%B6%E7%89%88%EF%BC%89)| Suitable for translating RPG, RenPY, Krkr games, and performing embedded translation |
      |[Paratranz Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Paratranz)| Suitable for translating MODs for various large-scale games |
      |[StevExtraction Tutorial](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90StevExtraction)| Suitable for translating RPGmakerMZ/MZ games |
      |[Unity Translation Tutorial](https://zhuanlan.zhihu.com/p/1894065679927313655)| Suitable for translating Unity games |
      |[Comprehensive Game Translation Tutorial](https://www.notion.so/AI-1d43d31f89b280f6bd61e12580652ce5?pvs=4)| Suitable for translating various games and creating high-quality embedded patches |

</details>

</details>

---

<details>
<summary>
  
## ⚙️ Feature Details
</summary>

<details>
<summary>
  
### ⚙️ Settings
</summary>

-   [Features - Interface Management](https://github.com/NEKOparapa/AiNiee/wiki/%E5%8A%9F%E8%83%BD%E2%80%90%E6%8E%A5%E5%8F%A3%E7%AE%A1%E7%90%86)

</details>

<details>
<summary> 

### 📄 Table Descriptions
</summary>

-   [Table - AI Terminology](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E6%9C%AF%E8%AF%AD%E8%A1%A8%E4%BB%8B%E7%BB%8D)
-   [Table - AI Forbidden Words](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E7%A6%81%E7%BF%BB%E8%A1%A8%E4%BB%8B%E7%BB%8D)
-   [Table - Text Replacement](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90%E6%96%87%E6%9C%AC%E6%9B%BF%E6%8D%A2%E4%BB%8B%E7%BB%8D)

</details>

<details>
<summary>
  
### 🔌 Plugin Details
</summary>

-   [Plugin - Language Filter](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90LanguageFilter)
-   [Plugin - Text Normalizer](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90TextNormalizer)

</details>

<details>
<summary> 

### ℹ️ Additional Information
</summary>

*   ` Multiple Key Rotation`
    >   If you want to use multiple keys to share the consumption pressure and speed up the translation, please use keys of the same type of account, and add an English comma between each key, without line breaks. For example: key1,key2,key3

*   ` Batch File Translation`
    >   Just put all the files that need to be translated in the input folder, and it also supports multiple folder structures

*   ` Configuration Migration`
    >   Configuration information will be stored in the config.json of resource. You can copy it to the resource of the new version when downloading the new version.

</details>
</details>

---

<details>
<summary>

## 🤝 Contributing
</summary>

*   **`Developing Enhanced Plugins`**: Develop more powerful functional plugins based on the [Plugin Writing Guide](https://github.com/NEKOparapa/AiNiee/blob/main/PluginScripts/README.md).

*   **`Improving or Adding Support Files`**: Requires certain code programming ability, pull the source code to make improvements. The specific file reading code is in the ModuleFolders\\FileReader and FileOutputer folders. [Reader/Writer System Writing Guide](https://github.com/NEKOparapa/AiNiee/blob/main/ModuleFolders/FileAccessor/README.md). UI support in UserInterface\Setting of ProjectSettingsPage.

*   **`Improving the Regular Expression Library`**: The completeness of the regular expression library will greatly help the work of game embedding, and benefit the next game translation work and benefit other translation users. The regular expression library is in the [Resource\Regex](https://github.com/NEKOparapa/AiNiee/blob/main/Resource/Regex/regex.json) folder

*   **`Improving UI Translation`**: The UI text of multi-language interfaces may not be accurately translated, you can submit your modification suggestions, or modify it directly. Localization text is in the [Resource\Localization](https://github.com/NEKOparapa/AiNiee/tree/main/Resource/Localization) folder

</details>

---

<details>
<summary>

## 📜 Special Notice
</summary>

AiNiee has evolved through the continuous personal research, user feedback, and contributions from developers since the project's inception.
This has been a process of exploration, continuous improvement, and joint construction over the past two years, which has formed the relatively mature and complete AI translation system of AiNiee.
When using and learning, please respect the spirit of open source, indicate the source project, and don't forget to give the project a star.

This AI translation tool is only for personal legal use. Any behavior of directly or indirectly engaging in illegal profit-making activities using this tool does not belong to the scope of authorization, nor is it subject to any support or recognition.

*   **`交流群`**: QQ Group (Mainly Active, Answer: github): 821624890, Backup TG Group: https://t.me/+JVHbDSGo8SI2Njhl,

</details>

---

## 💖 Sponsor
[![赞助](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/徽章.png)](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/赞赏码.png)