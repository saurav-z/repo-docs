<!-- # BabelDOC: Your Go-To PDF Translator -->

<div align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
      <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
    </picture>
</div>

<div align="center">
    <p>
      <!-- PyPI -->
      <a href="https://pypi.org/project/BabelDOC/">
        <img src="https://img.shields.io/pypi/v/BabelDOC"></a>
      <a href="https://pepy.tech/projects/BabelDOC">
        <img src="https://static.pepy.tech/badge/BabelDOC"></a>
      <!-- License -->
      <a href="./LICENSE">
        <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC"></a>
      <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
        <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white"></a>
    </p>
    <a href="https://trendshift.io/repositories/13358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

## BabelDOC: Effortlessly Translate Scientific PDFs

BabelDOC is a powerful Python library designed to translate scientific PDF papers, offering both online and self-deployment options for your translation needs.  Check out the [original repository](https://github.com/funstory-ai/BabelDOC) for more details.

**Key Features:**

*   🚀 **Fast Translation:**  Quickly translate scientific documents.
*   🌐 **Online Service:**  Try the beta version on [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) for free (1000 pages/month).
*   💻 **Self-Deployment:** Utilize [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosted solutions.
*   ⚙️ **Command Line Interface & Python API:**  Translate PDFs directly or integrate BabelDOC into your projects.
*   📚 **Zotero Integration:** Seamlessly integrate with Zotero via the [immersive-translate/zotero-immersivetranslate](https://github.com/immersive-translate/zotero-immersivetranslate) or [guaguastandup/zotero-pdf2zh](https://github.com/guaguastandup/zotero-pdf2zh) plugins.
*   🌍 **Multi-Language Support:**  Translate English to Chinese. Basic English target language support has been added.

<div align="center">
    <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## Getting Started

### Installation

We recommend installing BabelDOC using [uv](https://github.com/astral-sh/uv).

**Install using `uv` tool:**

1.  Install `uv` following the instructions:  [uv installation](https://github.com/astral-sh/uv#installation)
2.  Install BabelDOC:

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```
3.  Use the `babeldoc` command:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    ```

**Install from Source:**

1.  Follow the `uv` installation instructions:  [uv installation](https://github.com/astral-sh/uv#installation)
2.  Clone the repository:

    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    ```
3.  Install dependencies and run:

    ```bash
    uv run babeldoc --help
    ```
4. Use the `uv run babeldoc` command:

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

## Advanced Options
For detailed CLI options, including language, PDF processing, translation service, OpenAI settings, glossary options, and output control, please refer to the original [README](https://github.com/funstory-ai/BabelDOC).

## Python API

The `high_level.do_translate_async_stream` function within [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next) is the currently recommended method for integrating BabelDOC into Python projects.

## Contributing

Contribute to BabelDOC and help improve the translation experience!  See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide for more information.  Active contributors are eligible for monthly Pro membership redemption codes to [Immersive Translation](https://immersivetranslate.com) .

## Acknowledgements

*   [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
*   [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
*   [pdfminer](https://github.com/pdfminer/pdfminer.six)
*   [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
*   [Asynchronize](https://github.com/multimeric/Asynchronize/tree/master?tab=readme-ov-file)
*   [PriorityThreadPoolExecutor](https://github.com/oleglpts/PriorityThreadPoolExecutor)

## Star History

<a href="https://star-history.com/#funstory-ai/babeldoc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date"/>
 </picture>
</a>