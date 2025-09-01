---
title: BabelDOC: Effortlessly Translate PDF Scientific Papers 
description: BabelDOC is a powerful library for translating PDF scientific papers and comparing bilingual text, offering online and self-hosted options. This library simplifies PDF translation using advanced AI, providing a command-line interface, a Python API, and integration with tools like Zotero.
keywords: PDF translation, scientific paper translator, bilingual comparison, PDF to Chinese, PDF to English, AI translation, document translation, BabelDOC, PDFMathTranslate
---

# BabelDOC: Translate PDF Scientific Papers with Ease

**Effortlessly translate and compare scientific papers with BabelDOC, a powerful library for PDF translation.**  [Explore the BabelDOC repository](https://github.com/funstory-ai/BabelDOC)

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
  <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
</picture>

<p>
  <a href="https://pypi.org/project/BabelDOC/">
    <img src="https://img.shields.io/pypi/v/BabelDOC"></a>
  <a href="https://pepy.tech/projects/BabelDOC">
    <img src="https://static.pepy.tech/badge/BabelDOC"></a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC"></a>
  <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
    <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white"></a>
</p>

<a href="https://trendshift.io/repositories/13358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

**Key Features:**

*   🚀 **Accurate Translations:** Leverage advanced AI for high-quality PDF translations.
*   🌐 **Flexible Deployment:** Use our online service or self-deploy with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).
*   💻 **Command-Line Interface:** Easily translate PDFs from your terminal.
*   🐍 **Python API:** Integrate BabelDOC into your Python projects.
*   📚 **Zotero Integration:**  Seamlessly translate papers within your Zotero workflow. (via Immersive Translate Pro or PDFMathTranslate)
*   🗣️ **Multilingual Support:** Translates from English to Chinese, and supports English as a target language.

## Online Service

*   **Immersive Translate - BabelDOC:** Experience the beta version with 1000 free pages per month.

## Getting Started

### Installation

**Using uv (Recommended):**

1.  Install [uv](https://github.com/astral-sh/uv#installation).
2.  Install BabelDOC:

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

3.  Use the `babeldoc` command:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    ```

**From Source:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    ```
2.  Install dependencies and run BabelDOC:
    ```bash
    uv run babeldoc --help
    ```
3.  Use the `uv run babeldoc` command:

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

## Advanced Options

### Language Options

*   `--lang-in`, `-li`: Source language code (default: en)
*   `--lang-out`, `-lo`: Target language code (default: zh)

### PDF Processing Options

*   `--files`: Input PDF documents.
*   `--pages`, `-p`: Specify pages to translate.
*   ... (See the original README for a comprehensive list)

### Translation Service Options

*   `--openai`: Use OpenAI for translation (default: False)
*   `--openai-model`: OpenAI model to use (default: gpt-4o-mini)
*   `--openai-base-url`: Base URL for OpenAI API
*   ... (See the original README for a comprehensive list)

### Glossary Options

*   `--glossary-files`: Use a glossary for specialized terminology.

### Output Control

*   `--output`, `-o`: Output directory.
*   `--debug`: Enable debug logging.

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package.
*   `--restore-offline-assets`: Restore an offline assets package.

## Python API

>   The current recommended way to call BabelDOC in Python is to call the `high_level.do_translate_async_stream` function of [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).
>   **All APIs of BabelDOC should be considered as internal APIs, and any direct use of BabelDOC is not supported.**

## Roadmap

*   Line support
*   Table support
*   Cross-page/cross-column paragraph support
*   More advanced typesetting features
*   Outline support
*   ...

## Known Issues

*   Parsing errors in author and reference sections.
*   Line and drop cap support are missing.
*   Large pages may be skipped.

## Contribute

Contribute to BabelDOC! See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.
Contributors receive monthly Pro membership redemption codes for Immersive Translation. See [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md) for details.

## Acknowledgements

(List of acknowledgements)

<h2 id="star_hist">Star History</h2>

<a href="https://star-history.com/#funstory-ai/babeldoc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date"/>
 </picture>
</a>

## Important Notes for `--auto-enable-ocr-workaround`

(See the original README for details.)