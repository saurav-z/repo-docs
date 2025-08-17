---
title: BabelDOC: Effortlessly Translate PDF Scientific Papers
description: BabelDOC is a powerful Python library and command-line tool for translating PDF scientific papers. Translate documents with ease using OpenAI or self-deployable options.
keywords: PDF translation, scientific paper translation, OpenAI, PDFMathTranslate, bilingual PDF, Python library, document translator
---

<div align="center">
<!-- <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-banner.png" width="320px"  alt="YADT"/> -->

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
  <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
</picture>

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

## BabelDOC: Translate Your PDFs with Ease

**BabelDOC is your go-to solution for translating scientific papers and other PDF documents, offering both online and self-deployment options for seamless translation.**  [Explore the original repository](https://github.com/funstory-ai/BabelDOC).

**Key Features:**

*   **Online Service:** Beta version available at [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) with 1000 free pages per month.
*   **Self-Deployment:** Utilize [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting and a WebUI.
*   **Command-Line Interface (CLI):** Translate PDFs directly from your terminal.
*   **Python API:** Integrate BabelDOC into your Python projects.
*   **Bilingual PDF Output:** Generate translated PDFs with original and translated text side-by-side.
*   **OpenAI Integration:** Leverage the power of OpenAI for high-quality translations.
*   **Glossary Support:** Improve translation accuracy with custom glossaries.
*   **Offline Assets:** Manage offline assets for environments without internet access.

## Getting Started

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) for environment management.

1.  **Install uv:** Follow the [uv installation instructions](https://github.com/astral-sh/uv#installation).
2.  **Install BabelDOC:**

    *   **From PyPI:**
        ```bash
        uv tool install --python 3.12 BabelDOC
        babeldoc --help
        ```
    *   **From Source:**
        ```bash
        git clone https://github.com/funstory-ai/BabelDOC
        cd BabelDOC
        uv run babeldoc --help
        ```

### Basic Usage

*   **Translate a PDF using OpenAI:**

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    ```

*   **Translate multiple PDFs:**

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```

## Advanced Options

**[See complete documentation in the original repository.](https://github.com/funstory-ai/BabelDOC)**

**[Note:** The command line interface is mainly for debugging. The recommended way is to use the online service or self-deploy using PDFMathTranslate 2.0.

### Language Options

*   `--lang-in`, `-li`: Source language (default: `en`)
*   `--lang-out`, `-lo`: Target language (default: `zh`)

### PDF Processing Options

*   `--files`: Input PDF file paths.
*   `--pages`, `-p`: Specify pages to translate (e.g., "1,2,1-,-3,3-5"). If not set, translate all pages
*   `--split-short-lines`: Split short lines into different paragraphs
*   `--short-line-split-factor`: Split threshold factor (default: 0.8)
*   `--skip-clean`: Skip PDF cleaning step
*   `--dual-translate-first`: Put translated pages first in dual PDF mode (default: original pages first)
*   `--disable-rich-text-translate`: Disable rich text translation
*   `--enhance-compatibility`: Enable all compatibility enhancement options
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF.
*   `--watermark-output-mode`: Control watermark output mode: 'watermarked' (default), 'no_watermark', or 'both'.
*   `--max-pages-per-part`: Maximum pages per part for split translation.
*   `--translate-table-text`: Translate table text (experimental).
*   `--formular-font-pattern`: Font pattern for formula text.
*   `--formular-char-pattern`: Character pattern for formula text.
*   `--show-char-box`: Show character bounding boxes (debug).
*   `--skip-scanned-detection`: Skip scanned document detection.
*   `--ocr-workaround`: Use OCR workaround.
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround.
*   `--primary-font-family`: Override primary font family.
*   `--only-include-translated-page`: Only include translated pages in the output PDF.
*   `--rpc-doclayout`: RPC service host for document layout analysis.
*   `--working-dir`: Working directory for translation.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file.

### Translation Service Options

*   `--qps`: Queries Per Second limit (default: 4).
*   `--ignore-cache`: Force retranslation.
*   `--no-dual`: Do not output bilingual PDF.
*   `--no-mono`: Do not output monolingual PDF.
*   `--min-text-length`: Minimum text length to translate (default: 5).
*   `--openai`: Use OpenAI for translation (default: False).
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint.
*   `--pool-max-workers`: Maximum worker threads.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model (default: `gpt-4o-mini`).
*   `--openai-base-url`: OpenAI API base URL.
*   `--openai-api-key`: OpenAI API key.

### Glossary Options

*   `--glossary-files`: Comma-separated paths to glossary CSV files.

### Output Control

*   `--output`, `-o`: Output directory.
*   `--debug`: Enable debug logging.
*   `--report-interval`: Progress report interval (default: 0.1).

### General Options

*   `--warmup`: Download assets and exit.

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package.
*   `--restore-offline-assets`: Restore an offline assets package.

### Configuration File

*   `--config`, `-c`: Configuration file path (TOML format).  See the original README for example configurations.

## Python API

**[Refer to the original repository for the Python API example in `babeldoc/main.py`.](https://github.com/funstory-ai/yadt/blob/main/babeldoc/main.py)**

**Important Notes:**

1.  Make sure to call `babeldoc.format.pdf.high_level.init()` before using the API
2.  Validate input parameters.
3.  For offline assets management: use `babeldoc.assets.assets.generate_offline_assets_package()` and `babeldoc.assets.assets.restore_offline_assets_package()`.

## Background

BabelDOC builds upon prior work in PDF parsing and translation, aiming to provide a standardized and extensible pipeline.  It focuses on parsing, rendering, and offering an intermediate representation to preserve structure.  This allows flexibility for adding new models, OCR engines, and renderers.

## Roadmap

*   Add line support.
*   Add table support.
*   Add cross-page/cross-column paragraph support.
*   More advanced typesetting features.
*   Outline support.
*   And More

## Versioning

This project uses [Semantic Versioning](https://semver.org/) and [Pride Versioning](https://pridever.org/) ("0.MAJOR.MINOR").

## Known Issues

*   Parsing errors in author and reference sections.
*   Lines are not supported.
*   Drop caps are not supported.
*   Large pages may be skipped.

## How to Contribute

Contributions are welcome!  See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

## Acknowledgements

*   [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
*   [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
*   [pdfminer](https://github.com/pdfminer/pdfminer.six)
*   [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
*   [Asynchronize](https://github.com/multimeric/Asynchronize/tree/master?tab=readme-ov-file)
*   [PriorityThreadPoolExecutor](https://github.com/oleglpts/PriorityThreadPoolExecutor)

<h2 id="star_hist">Star History</h2>

<a href="https://star-history.com/#funstory-ai/babeldoc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date"/>
 </picture>
</a>