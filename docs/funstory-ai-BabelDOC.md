<!-- # BabelDOC: Scientific PDF Translation and Bilingual Comparison -->

<div align="center">
    <a href="https://github.com/funstory-ai/BabelDOC">
        <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC" />
    </a>
</div>

<div align="center">
    <!-- PyPI -->
    <a href="https://pypi.org/project/BabelDOC/">
        <img src="https://img.shields.io/pypi/v/BabelDOC">
    </a>
    <a href="https://pepy.tech/projects/BabelDOC">
        <img src="https://static.pepy.tech/badge/BabelDOC">
    </a>
    <!-- License -->
    <a href="./LICENSE">
        <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC">
    </a>
    <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
        <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white">
    </a>
</div>

<a href="https://trendshift.io/repositories/13358" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55" />
</a>

## BabelDOC: Effortlessly Translate and Compare Scientific PDFs

BabelDOC is your go-to solution for translating scientific PDF papers and comparing the original with the translated version, offering both online services and self-deployment options. Visit the [original BabelDOC repository](https://github.com/funstory-ai/BabelDOC) for more information.

**Key Features:**

*   **Accurate Translation:** Utilizes advanced techniques, including LLMs and glossaries, to provide high-quality translations.
*   **Bilingual Comparison:** Generates side-by-side comparisons of original and translated PDFs for easy review.
*   **Online Service:** Access the Beta version of the online service, [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/), with 1000 free pages per month.
*   **Self-Deployment Options:** Integrate with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting.
*   **Command Line Interface (CLI):** Translate PDFs directly from your terminal.
*   **Python API:** Embed BabelDOC's capabilities into your own Python projects.

## Getting Started

### Installation

BabelDOC can be installed using [uv](https://github.com/astral-sh/uv). Make sure you have `uv` installed and configured correctly.

1.  **Using `uv`:**

    *   Install BabelDOC using:
        ```bash
        uv tool install --python 3.12 BabelDOC
        ```
    *   Use the `babeldoc` command:
        ```bash
        babeldoc --help
        ```

2.  **From Source:**

    *   Clone the repository:
        ```bash
        git clone https://github.com/funstory-ai/BabelDOC
        cd BabelDOC
        ```
    *   Run BabelDOC using `uv`:
        ```bash
        uv run babeldoc --help
        ```

    *   For example:
        ```bash
        uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
        ```

    *   Remember to replace `"your-api-key-here"` with your actual OpenAI API key if using OpenAI for translation.

### Usage
    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf

    # multiple files
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```

## Advanced Options

Explore a wide range of options for customizing your PDF translation, including language selection, PDF processing, translation service configurations (OpenAI), glossary integration, output control, and offline assets management.

### Language Options
*   `--lang-in`, `-li`: Source language code (default: en)
*   `--lang-out`, `-lo`: Target language code (default: zh)

### PDF Processing Options
*   `--files`: PDF documents
*   `--pages`, `-p`: Specify pages
*   `--split-short-lines`: Force split short lines
*   `--short-line-split-factor`: Split threshold factor (default: 0.8)
*   `--skip-clean`: Skip PDF cleaning step
*   `--dual-translate-first`: Translated pages first in dual PDF mode (default: original pages first)
*   `--disable-rich-text-translate`: Disable rich text translation
*   `--enhance-compatibility`: Enable all compatibility enhancement options
*   `--use-alternating-pages-dual`: Use alternating pages mode
*   `--watermark-output-mode`: 'watermarked', 'no_watermark', 'both'
*   `--max-pages-per-part`: Maximum pages per part for split translation
*   `--translate-table-text`: Translate table text (experimental, default: False)
*   `--formular-font-pattern`: Font pattern to identify formula text (default: None)
*   `--formular-char-pattern`: Character pattern to identify formula text (default: None)
*   `--show-char-box`: Show character bounding boxes (debug only, default: False)
*   `--skip-scanned-detection`: Skip scanned document detection (default: False).
*   `--ocr-workaround`: Use OCR workaround (default: False).
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False).
*   `--primary-font-family`: Override primary font family
*   `--only-include-translated-page`: Only include translated pages (default: False)
*   `--merge-alternating-line-numbers`: Enable post-processing (default: off)
*   `--skip-form-render`: Skip form rendering (default: False)
*   `--skip-curve-render`: Skip curve rendering (default: False)
*   `--only-parse-generate-pdf`: Only parse PDF and generate output PDF (default: False)
*   `--remove-non-formula-lines`: Remove non-formula lines (default: False)
*   `--non-formula-line-iou-threshold`: IoU threshold (default: 0.9)
*   `--figure-table-protection-threshold`: IoU threshold (default: 0.9)
*   `--rpc-doclayout`: RPC service host for document layout analysis (default: None)
*   `--working-dir`: Working directory
*   `--no-auto-extract-glossary`: Disable automatic term extraction (default: enabled)
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary

### Translation Service Options
*   `--qps`: Queries Per Second limit (default: 4)
*   `--ignore-cache`: Force retranslation
*   `--no-dual`: Do not output bilingual PDF
*   `--no-mono`: Do not output monolingual PDF
*   `--min-text-length`: Minimum text length to translate (default: 5)
*   `--openai`: Use OpenAI for translation (default: False)
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation. (default: False)
*   `--pool-max-workers`: Maximum worker threads for internal task processing pools. (defaults to QPS)
*   `--no-auto-extract-glossary`: Disable automatic term extraction (default: enabled)

### OpenAI Specific Options
*   `--openai-model`: OpenAI model (default: gpt-4o-mini)
*   `--openai-base-url`: OpenAI API base URL
*   `--openai-api-key`: OpenAI API key

### Glossary Options
*   `--glossary-files`: Comma-separated paths to glossary CSV files

### Output Control
*   `--output`, `-o`: Output directory (default: current)
*   `--debug`: Enable debug logging
*   `--report-interval`: Progress report interval (default: 0.1)

### General Options
*   `--warmup`: Only download and verify assets, then exit (default: False)

### Offline Assets Management
*   `--generate-offline-assets`: Generate offline assets package.
*   `--restore-offline-assets`: Restore offline assets package.

### Configuration File
*   `--config`, `-c`: Configuration file path (TOML format).
    See the original README for example.

## Python API

*   For detailed API information, please refer to the `high_level.do_translate_async_stream` function from `pdf2zh next`.  **Direct use of BabelDOC APIs is not supported.**

## Background

BabelDOC is built upon an understanding of PDF document structures, breaking down the processes into parsing and rendering stages.  It provides an intermediate representation, which can be rendered into different formats.  The pipeline is plugin-based, allowing for flexible customization.

## Roadmap

*   Add line support
*   Add table support
*   Add cross-page/cross-column paragraph support
*   More advanced typesetting features
*   Outline support
*   ...

## Versioning

BabelDOC uses Semantic Versioning combined with Pride Versioning, with a version number format of "0.MAJOR.MINOR."

## Known Issues

*   Parsing errors in author and reference sections.
*   Line and drop cap support are missing.
*   Large pages may be skipped.

## How to Contribute

Contribute to the project by following the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

## Acknowledgements

*   PDFMathTranslate
*   DocLayout-YOLO
*   pdfminer
*   PyMuPDF
*   Asynchronize
*   PriorityThreadPoolExecutor

## Star History

```html
<a href="https://star-history.com/#funstory-ai/babeldoc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date"/>
 </picture>
</a>
```