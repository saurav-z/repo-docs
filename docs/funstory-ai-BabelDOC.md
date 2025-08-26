# BabelDOC: Effortlessly Translate PDF Scientific Papers 🚀

**Tired of struggling with complex scientific papers in foreign languages?** BabelDOC is a powerful, open-source library designed to translate PDF scientific papers, providing both online and self-deployment options.  Unlock knowledge and accelerate your research with easy-to-use PDF translation!

[![PyPI version](https://img.shields.io/pypi/v/BabelDOC)](https://pypi.org/project/BabelDOC/)
[![Downloads](https://static.pepy.tech/badge/BabelDOC)](https://pepy.tech/projects/BabelDOC)
[![License](https://img.shields.io/github/license/funstory-ai/BabelDOC)](./LICENSE)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white)](https://t.me/+Z9_SgnxmsmA5NzBl)

[<img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>](https://trendshift.io/repositories/13358)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
  <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
</picture>

**Get started with BabelDOC today!**  [See the original repository](https://github.com/funstory-ai/BabelDOC).

## Key Features

*   **PDF Translation:**  Accurately translate scientific papers.
*   **Online Service:** Beta version available at [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) with 1000 free pages per month.
*   **Self-Deployment:** Integrate with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for custom WebUI and more.
*   **Command Line Interface (CLI):** Translate documents directly from your terminal.
*   **Python API:** Embed translation functionality into your own Python applications.
*   **Zotero Integration**: Seamlessly integrate BabelDOC with Zotero.

## Preview

<div align="center">
<img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## Getting Started

### Installation using `uv` (Recommended)

1.  Install `uv`: Refer to [uv installation](https://github.com/astral-sh/uv#installation) and set up your `PATH`.
2.  Install BabelDOC:

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```
3.  Run the command:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    # multiple files
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```

### Install from Source

1.  Install `uv`: Follow the [uv installation](https://github.com/astral-sh/uv#installation) guide.
2.  Clone the Repository & Install:

    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    uv run babeldoc --help
    ```
3.  Run BabelDOC:

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"

    # multiple files
    uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

## Advanced Options & Configuration

BabelDOC offers a wide range of options for customization.  Detailed configurations are available for:

*   **Language Options**
*   **PDF Processing Options**
*   **Translation Service Options**
*   **OpenAI Specific Options**
*   **Glossary Options**
*   **Output Control**
*   **Offline Assets Management**
*   **Configuration File** (TOML format)

For detailed instructions and options, refer to the original README.

## Python API

Refer to the [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next) documentation for using the `high_level.do_translate_async_stream` function.

## Background & Motivation

BabelDOC addresses the need for a robust and customizable PDF translation pipeline, building upon existing projects and research in PDF parsing and document structure analysis.

## Roadmap

Future development includes:

*   Line Support
*   Table Support
*   Cross-page/column paragraph support
*   Advanced typesetting features
*   Outline Support
*   And more!

## Versioning

This project uses a combination of [Semantic Versioning](https://semver.org/) and [Pride Versioning](https://pridever.org/). The version number format is: "0.MAJOR.MINOR".

## Known Issues

*   Parsing errors in the author and reference sections
*   Lines are not supported.
*   Drop caps are not supported.
*   Large pages will be skipped.

## How to Contribute

Contributions are welcome!  See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide for details.

## Acknowledgements

*   [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
*   [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
*   [pdfminer](https://github.com/pdfminer/pdfminer.six)
*   [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
*   [Asynchronize](https://github.com/multimeric/Asynchronize/tree/master?tab=readme-ov-file)
*   [PriorityThreadPoolExecutor](https://github.com/oleglpts/PriorityThreadPoolExecutor)

## Star History
```html
<h2 id="star_hist">Star History</h2>

<a href="https://star-history.com/#funstory-ai/babeldoc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date"/>
 </picture>
</a>
```

## Important Interaction Note for `--auto-enable-ocr-workaround`

(See original README for details.)

---