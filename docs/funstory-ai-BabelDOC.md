<!-- # BabelDOC: Scientific PDF Translation & Bilingual Comparison -->

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

## BabelDOC: Translate and Compare Scientific PDFs with AI

BabelDOC is a powerful Python library designed for translating and comparing scientific PDF documents, providing both online and self-hosted options.  Explore the [BabelDOC GitHub Repository](https://github.com/funstory-ai/BabelDOC) for more details.

**Key Features:**

*   **PDF Translation:**  Translate scientific papers from English to Chinese (and increasingly, other languages) using advanced AI models.
*   **Bilingual Comparison:** Generate dual-language PDFs for side-by-side comparison of original and translated text.
*   **Command Line Interface (CLI):** Easily translate PDFs with a simple command-line tool.
*   **Python API:** Integrate BabelDOC's translation capabilities into your own Python applications.
*   **Online Service:**  Try the beta version on [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) (1000 free pages per month).
*   **Self-Deployment:**  Leverage [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting and a WebUI.
*   **Glossary Support:** Incorporate custom glossaries for accurate domain-specific translations.
*   **Offline Assets:** Support for offline operation with assets packaged for environments without internet access.

## Getting Started

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) for managing your environment.

1.  **Install uv:** Follow the [uv installation instructions](https://github.com/astral-sh/uv#installation).
2.  **Install BabelDOC from PyPI:**

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

3.  **Use the `babeldoc` command**:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    # multiple files
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```

### Installation from Source

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    ```

2.  **Install dependencies and run BabelDOC:**

    ```bash
    uv run babeldoc --help
    ```

3.  **Use the `uv run babeldoc` command:**

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    # multiple files
    uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

## Advanced Options

BabelDOC offers several advanced options to customize your translation process.  For debugging, use the CLI. End users should utilize [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) or [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

### Language Options

*   `--lang-in`, `-li`: Source language code (default: en)
*   `--lang-out`, `-lo`: Target language code (default: zh)

### PDF Processing Options

*   `--files`: Input PDF file paths
*   `--pages`, `-p`: Specify pages to translate (e.g., "1,2,1-,-3,3-5"). If not set, translate all pages
*   `--split-short-lines`: Force split short lines into different paragraphs (may cause poor typesetting & bugs)
*   `--short-line-split-factor`: Split threshold factor (default: 0.8)
*   `--skip-clean`: Skip PDF cleaning step
*   `--dual-translate-first`: Put translated pages first in dual PDF mode (default: original pages first)
*   `--disable-rich-text-translate`: Disable rich text translation
*   `--enhance-compatibility`: Enable all compatibility enhancement options
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF.
*   `--watermark-output-mode`: Control watermark output mode
*   `--max-pages-per-part`: Maximum number of pages per part for split translation
*   `--translate-table-text`: Translate table text (experimental, default: False)
*   `--formular-font-pattern`: Font pattern to identify formula text
*   `--formular-char-pattern`: Character pattern to identify formula text
*   `--show-char-box`: Show character bounding boxes (debug only, default: False)
*   `--skip-scanned-detection`: Skip scanned document detection (default: False).
*   `--ocr-workaround`: Use OCR workaround (default: False).
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False).
*   `--primary-font-family`: Override primary font family for translated text.
*   `--only-include-translated-page`: Only include translated pages in the output PDF.

### Translation Service Options

*   `--qps`: QPS (Queries Per Second) limit for translation service (default: 4)
*   `--ignore-cache`: Ignore translation cache and force retranslation
*   `--no-dual`: Do not output bilingual PDF files
*   `--no-mono`: Do not output monolingual PDF files
*   `--min-text-length`: Minimum text length to translate (default: 5)
*   `--openai`: Use OpenAI for translation (default: False)
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation. (Currently not recommended, it may affect translation quality, default: False)
*   `--pool-max-workers`: Maximum number of worker threads for internal task processing pools. If not specified, defaults to QPS value. This parameter directly sets the worker count, replacing previous QPS-based dynamic calculations.
*   `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model to use (default: gpt-4o-mini)
*   `--openai-base-url`: Base URL for OpenAI API
*   `--openai-api-key`: API key for OpenAI service

### Glossary Options

*   `--glossary-files`: Comma-separated paths to glossary CSV files.

### Output Control

*   `--output`, `-o`: Output directory for translated files.
*   `--debug`: Enable debug logging
*   `--report-interval`: Progress report interval in seconds (default: 0.1)

### General Options

*   `--warmup`: Only download and verify required assets then exit (default: False)

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package.
*   `--restore-offline-assets`: Restore an offline assets package.

### Configuration File

*   `--config`, `-c`: Configuration file path (TOML format).

## Python API

While awaiting the release of pdf2zh 2.0, you can use BabelDOC's Python API.  Refer to [main.py](https://github.com/funstory-ai/yadt/blob/main/babeldoc/main.py) for usage examples.

## Background & Motivation

BabelDOC aims to streamline scientific document translation, addressing the challenges of parsing and rendering complex PDF structures.  It offers a flexible, plugin-based pipeline for translation, building upon existing solutions like PDFMathTranslate and other projects to deliver a more efficient and accurate translation process.

## Roadmap

*   Line Support
*   Table Support
*   Cross-Page/Cross-Column Paragraph Support
*   Advanced Typesetting Features
*   Outline Support
*   ...

## Versioning

BabelDOC uses Semantic Versioning and Pride Versioning. The version number format is: "0.MAJOR.MINOR".

## Known Issues

*   Parsing errors in the author and reference sections.
*   Lines are not supported.
*   Does not support drop caps.
*   Large pages may be skipped.

## Contribute

We welcome contributions! See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

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

> [!WARNING]
> **Important Interaction Note for `--auto-enable-ocr-workaround`:**
>
> When `--auto-enable-ocr-workaround` is set to `true` (either via command line or config file):
>
> 1.  During the initial setup, the values for `ocr_workaround` and `skip_scanned_detection` will be forced to `false` by `TranslationConfig`, regardless of whether you also set `--ocr-workaround` or `--skip-scanned-detection` flags.
> 2.  Then, during the scanned document detection phase (`DetectScannedFile` stage):
>     *   If the document is identified as heavily scanned (e.g., >80% scanned pages) AND `auto_enable_ocr_workaround` is `true` (i.e., `translation_config.auto_enable_ocr_workaround` is true), the system will then attempt to set both `ocr_workaround` to `true` and `skip_scanned_detection` to `true`.
>
> This means that `--auto-enable-ocr-workaround` effectively gives the system control to enable OCR processing for scanned documents, potentially overriding manual settings for `--ocr-workaround` and `--skip_scanned_detection` based on its detection results. If the document is *not* detected as heavily scanned, then the initial `false` values for `ocr_workaround` and `skip_scanned_detection` (forced by `--auto-enable-ocr-workaround` at the `TranslationConfig` initialization stage) will remain in effect unless changed by other logic.