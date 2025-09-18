<div align="center">
  <a href="https://github.com/NEKOparapa/AiNiee">
    <img src="https://github.com/NEKOparapa/AiNiee-chatgpt/blob/main/Example%20image/logo.png" width=60%>
  </a>
</div>

<div align="center">
  <a href="README_EN.md">English</a> | 简体中文
</div>

---

## Unleash the Power of AI Translation with AiNiee!

AiNiee is a versatile AI-powered translation tool designed for seamless and efficient translation of various text formats, including games, books, subtitles, and documents.  **[Explore AiNiee on GitHub](https://github.com/NEKOparapa/AiNiee)**

---

## Key Features:

*   **Comprehensive Format Support:**
    *   🎮 **Game Translation:** Extensive support for popular game text extraction tools like Mtool, Translator++, ParaTranzr, Renpy, VNText, and SExtractor.
    *   📚 **Diverse File Types:** Handles I18Next data, Epub/TXT ebooks, Srt/Vtt/Lrc subtitles, Word/PDF/MD documents, and more.

*   **Smart and Efficient:**
    *   🚀 **One-Click Operation:** Simply drag and drop your files; AiNiee automatically detects file types and languages, eliminating the need for manual configuration.
    *   ⏱️ **Blazing Fast Translations:** Get your translated content in a matter of minutes, freeing up your time.

*   **Optimized for Quality:**
    *   🎯 **Advanced Techniques:** Leverages techniques like lightweight translation formats, chain-of-thought translation, AI terminology tables, and context association for superior accuracy and fluency in long texts.
    *   💎 **Enhanced Quality Control:** Includes features such as AI refinement, AI formatting, and terminology extraction, providing users with greater control over translation quality.

---

## Getting Started with AiNiee: Three Easy Steps

1.  **Configure Your API Interface:**
    *   <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第一步.png">

    *   Online APIs: Provide cost-effective, all-language support, with no GPU requirements.
        *   [DeepSeek API Setup Guide](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartDeepSeek)
        *   [Huo (Volcano Engine) API Setup Guide](https://github.com/NEKOparapa/AiNiee/wiki/QuickStartHuo) (Alternative if DeepSeek is unavailable)

2.  **Drag and Drop Your Files:**
    *   <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第二步.png">

    *   Place your source files into a dedicated folder and drag this folder into the designated area. Novels, subtitles, and documents can be translated directly. Games may require text extraction tools.

3.  **Start Translating:**
    *   <img src="https://raw.githubusercontent.com/NEKOparapa/AiNiee/main/Example%20image/三步走/第三步.png">

    *   Click the "Start" button and let AiNiee handle the rest.

    *   [Download AiNiee](https://github.com/NEKOparapa/AiNiee/releases)

---

<details>
<summary>
  
## Game Translation [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#游戏翻译)
</summary>

<details>
<summary>

### Tool Preparation
</summary>

*   **`📖 Game Text Extraction Tools`**

    | Tool Name | Description | Project Type |
    | :-----: | :-----: | :-----: |
    | [Mtool](https://afdian.com/p/d42dd1e234aa11eba42452540025c377) | Easy to use, recommended for beginners | Mtool export files |
    | [Translator++](https://dreamsavior.net/download/) | Complex, powerful, recommended for advanced users | T++ export files or Trans project files |
    | [ParaTranzr](https://paratranz.cn/projects) | Intermediate, powerful, recommended for advanced users | ParaTranzr export files |
    | [RenPy SDK](https://www.renpy.org/latest.html) | Intermediate, powerful, recommended for advanced users | renpy export files |

*   **`🧰 Local Model Running Tools`**

    | Tool Name | Description |
    | :-----: | :-----: |
    | [Sakura_Launcher_GUI](https://github.com/PiDanShouRouZhouXD/Sakura_Launcher_GUI) | Dedicated GUI launcher for Sakura models |
    | [LM Studio](https://lmstudio.ai/download) | A platform for local deployment of LLMs (Large Language Models), simplifies LLM usage and management. |
    | [ollama](https://ollama.com/) | Open-source, cross-platform large model tool |

</details>

<details>
<summary>

### Translation Tutorials
</summary>

*   **`📺 Game Translation Video Tutorials`**

    | Video Link | Description |
    | :-----: | :-----: |
    | [Mtool Tutorial (Bilibili)](https://www.bilibili.com/video/BV1h6421c7MA) | Recommended for first-time users |
    | [Translator++ Tutorial (Bilibili)](https://www.bilibili.com/video/BV1LgfoYzEaX/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for first-time users |
    | [Wolf Game Tutorial (Bilibili)](https://www.bilibili.com/video/BV1SnXbYiEjQ/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for first-time users |
    | [Name Reading Tutorial (Bilibili)](https://www.bilibili.com/video/BV1j1VyzqERD/?share_source=copy_web&vd_source=b0eede35fc5eaa5c382509c6040d6501) | Recommended for advanced translation |

*   **`🎫 Game Translation Text Tutorials`**

    | Video Link | Description |
    | :-----: | :-----: |
    | [Mtool Tutorial (GitHub Wiki)](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Mtool) | Suitable for beginners, quick translation of RPG, RenPY, Krkr games, external translation |
    | [Translator++ Tutorial (GitHub Wiki)](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Translator--%EF%BC%88%E5%B7%A5%E7%A8%8B%E6%96%87%E4%BB%B6%E7%89%88%EF%BC%89) | Suitable for translating RPG, RenPY, Krkr, and other games, in-embedded translation |
    | [Paratranz Tutorial (GitHub Wiki)](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90Paratranz) | Suitable for translating MODs of various large-scale games |
    | [StevExtraction Tutorial (GitHub Wiki)](https://github.com/NEKOparapa/AiNiee/wiki/%E6%B8%B8%E6%88%8F%E7%BF%BB%E8%AF%91%E2%80%90StevExtraction) | Suitable for translating RPGmakerMZ/MZ games |
    | [Unity Translation Tutorial (Zhihu)](https://zhuanlan.zhihu.com/p/1894065679927313655) | Suitable for translating Unity games |
    | [Comprehensive Game Translation Tutorial (Notion)](https://www.notion.so/AI-1d43d31f89b280f6bd61e12580652ce5?pvs=4) | Suitable for translating various games, creating high-quality embedded patches |

</details>

</details>

---

<details>
<summary>

## Functionality [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#功能说明)
</summary>

<details>
<summary>

### Settings Explanation
</summary>

-   [Features - API Management](https://github.com/NEKOparapa/AiNiee/wiki/%E5%8A%9F%E8%83%BD%E2%80%90%E6%8E%A5%E5%8F%A3%E7%AE%A1%E7%90%86)

</details>

<details>
<summary>

### Table Explanations
</summary>

-   [Table - AI Terminology Table](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E6%9C%AF%E8%AF%AD%E8%A1%A8%E4%BB%8B%E7%BB%8D)
-   [Table - AI Forbidden Translation Table](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90AI%E7%A6%81%E7%BF%BB%E8%A1%A8%E4%BB%8B%E7%BB%8D)
-   [Table - Text Replacement](https://github.com/NEKOparapa/AiNiee/wiki/%E8%A1%A8%E6%A0%BC%E2%80%90%E6%96%87%E6%9C%AC%E6%9B%BF%E6%8D%A2%E4%BB%8B%E7%BB%8D)

</details>

<details>
<summary>

### Plugin Explanations
</summary>

-   [Plugin - Language Filter](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90LanguageFilter)
-   [Plugin - Text Normalizer](https://github.com/NEKOparapa/AiNiee/wiki/%E6%8F%92%E4%BB%B6%E2%80%90TextNormalizer)

</details>

<details>
<summary>

### Other Explanations
</summary>

*   `Multiple Key Rotation`

    > If you want to use multiple keys to share the consumption pressure, please use keys of the same type, and add an English comma between each key, without line breaks. For example: key1,key2,key3

*   `Batch File Translation`

    > Just put all the files that need to be translated in the input folder, and it also supports multi-folder structures

*   `Configuration Migration`

    > Configuration information will be stored in the resource config.json, download the new version can copy it to the new version of the resource.

</details>

</details>

---

<details>
<summary>

## Contribution Guide [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#贡献指南)
</summary>

*   **`Develop Enhanced Plugins`**: Develop more powerful functional plugins according to the [Plugin Writing Guide](https://github.com/NEKOparapa/AiNiee/blob/main/PluginScripts/README.md)

*   **`Improve or Add Support Files`**: Requires some coding ability, pull the source code to improve. The specific code for reading the file is in the ModuleFolders\FileReader and FileOutputer folders. [Writing Guide for Reader and Writer System](https://github.com/NEKOparapa/AiNiee/blob/main/ModuleFolders/FileAccessor/README.md). UI support in the ProjectSettingsPage of UserInterface\Setting.

*   **`Improve the Regular Expression Library`**: The completeness of the regular expression library will greatly help the in-embedded work of the game, and benefit the next game translation work and benefit other translation users, the regular expression library is in the [Resource\Regex](https://github.com/NEKOparapa/AiNiee/blob/main/Resource/Regex/regex.json) folder

*   **`Improve Interface Translation`**: Multi-language interface UI text may not be accurately and appropriately translated, you can submit your modification suggestions, or modify directly. Localized text in the [Resource\Localization](https://github.com/NEKOparapa/AiNiee/tree/main/Resource/Localization) folder

</details>

---

<details>
<summary>

## Special Declaration [![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)](#特别声明)
</summary>

AiNiee's continuous development and iteration are based on individual research and development since the project was created, user feedback and suggestions, and the joint efforts and creation of senior developers' pull requests.

This is a process of continuous exploration, continuous improvement, and joint construction over the past two years, which has formed the relatively mature and complete AI translation system of AiNiee today.

Please respect the open-source spirit when using and learning, name the source project, and don't forget to give the project a star.

This AI translation tool is for personal legal use only. Any activity that uses this tool to engage in direct or indirect illegal profit-making is not within the scope of authorization and is not subject to any support or recognition.

*   **`♂ Exchange Group`**: QQ exchange group (main activity, answer: github): 821624890, spare TG group: https://t.me/+JVHbDSGo8SI2Njhl,

</details>

---

## Sponsorship💖

[![xxxx](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/徽章.png)](https://raw.githubusercontent.com/NEKOparapa/AiNiee-chatgpt/main/Example%20image/Sponsor/赞赏码.png)