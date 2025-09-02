# BabelDOC: Effortlessly Translate PDF Documents (Powered by AI)

**Translate your PDF documents with ease using BabelDOC, a powerful library offering both online and self-hosted solutions. Check out the original repo: [funstory-ai/BabelDOC](https://github.com/funstory-ai/BabelDOC)**

<div align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
      <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
    </picture>

    <p>
      <a href="https://pypi.org/project/BabelDOC/">
        <img src="https://img.shields.io/pypi/v/BabelDOC" alt="PyPI"></a>
      <a href="https://pepy.tech/projects/BabelDOC">
        <img src="https://static.pepy.tech/badge/BabelDOC" alt="Downloads"></a>
      <a href="./LICENSE">
        <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC" alt="License"></a>
      <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
        <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white" alt="Telegram"></a>
    </p>

    <a href="https://trendshift.io/repositories/13358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

## Key Features

*   **AI-Powered Translation:** Leverage the power of OpenAI to translate PDF documents.
*   **Online Service:** Access a user-friendly beta version via [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) with 1000 free pages/month.
*   **Self-Deployment:**  Deploy your own instance with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).
*   **Command-Line Interface (CLI):** Easily translate documents from the command line.
*   **Python API:** Integrate BabelDOC functionality into your Python projects.
*   **Dual-Language Output:** Generate bilingual PDFs for side-by-side comparison.
*   **Offline Assets Management:** Generate and restore offline assets packages.

## Getting Started

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) for installation.

1.  **Install `uv`:** Follow the instructions on the [uv installation page](https://github.com/astral-sh/uv#installation).
2.  **Install BabelDOC using `uv tool` (Recommended):**

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```
3.  **Use the `babeldoc` command:**

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf

    # Translate multiple files
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```

### Installation from Source

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    ```

2.  **Install dependencies and run `babeldoc`:**

    ```bash
    uv run babeldoc --help
    ```
3.  **Use the `uv run babeldoc` command:**

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"

    # Translate multiple files
    uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

## Advanced Options

**Note:** This CLI is primarily for debugging. For end-users, the [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) online service or [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-deployment are recommended.

### Language Options

*   `--lang-in`, `-li`: Source language code (default: en)
*   `--lang-out`, `-lo`: Target language code (default: zh)

### PDF Processing Options

*   `--files`: Input PDF files.
*   `--pages`, `-p`: Specify pages to translate.
*   `--split-short-lines`: Split short lines into paragraphs.
*   `--short-line-split-factor`: Split threshold factor (default: 0.8).
*   `--skip-clean`: Skip PDF cleaning.
*   `--dual-translate-first`: Translated pages first in dual PDF mode.
*   `--disable-rich-text-translate`: Disable rich text translation.
*   `--enhance-compatibility`: Enable all compatibility options.
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF.
*   `--watermark-output-mode`: Control watermark output mode.
*   `--max-pages-per-part`: Max pages per part for split translation.
*   `--translate-table-text`: Translate table text (experimental, default: False)
*   `--formular-font-pattern`: Font pattern to identify formula text (default: None)
*   `--formular-char-pattern`: Character pattern to identify formula text (default: None)
*   `--show-char-box`: Show character bounding boxes (debug only, default: False)
*   `--skip-scanned-detection`: Skip scanned document detection (default: False).
*   `--ocr-workaround`: Use OCR workaround (default: False).
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False).
*   `--primary-font-family`: Override primary font family.
*   `--only-include-translated-page`: Only include translated pages in the output PDF.
*   `--merge-alternating-line-numbers`: Enable post-processing to merge alternating line-number layouts.
*   `--skip-form-render`: Skip form rendering (default: False).
*   `--skip-curve-render`: Skip curve rendering (default: False).
*   `--only-parse-generate-pdf`: Only parse PDF and generate output PDF without translation (default: False).
*   `--remove-non-formula-lines`: Remove non-formula lines from paragraph areas (default: False).
*   `--non-formula-line-iou-threshold`: IoU threshold for detecting paragraph overlap when removing non-formula lines (default: 0.9).
*   `--figure-table-protection-threshold`: IoU threshold for protecting lines in figure/table areas when removing non-formula lines (default: 0.9).

*   `--rpc-doclayout`: RPC service host address for document layout analysis (default: None)
*   `--working-dir`: Working directory for translation.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file.

### Translation Service Options

*   `--qps`: QPS (Queries Per Second) limit for translation service (default: 4)
*   `--ignore-cache`: Ignore translation cache and force retranslation
*   `--no-dual`: Do not output bilingual PDF files
*   `--no-mono`: Do not output monolingual PDF files
*   `--min-text-length`: Minimum text length to translate (default: 5)
*   `--openai`: Use OpenAI for translation (default: False)
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation.
*   `--pool-max-workers`: Maximum number of worker threads.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model to use (default: gpt-4o-mini)
*   `--openai-base-url`: Base URL for OpenAI API
*   `--openai-api-key`: API key for OpenAI service

### Glossary Options

*   `--glossary-files`: Comma-separated paths to glossary CSV files.

### Output Control

*   `--output`, `-o`: Output directory.
*   `--debug`: Enable debug logging.
*   `--report-interval`: Progress report interval (default: 0.1).

### General Options

*   `--warmup`: Only download and verify assets then exit.

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package.
*   `--restore-offline-assets`: Restore an offline assets package.

### Configuration File

*   `--config`, `-c`: Configuration file path (TOML format).

#### Example Configuration (TOML)

```toml
[babeldoc]
# Basic settings
debug = true
lang-in = "en-US"
lang-out = "zh-CN"
qps = 10
output = "/path/to/output/dir"

# PDF processing options
split-short-lines = false
short-line-split-factor = 0.8
skip-clean = false
dual-translate-first = false
disable-rich-text-translate = false
use-alternating-pages-dual = false
watermark-output-mode = "watermarked"  # Choices: "watermarked", "no_watermark", "both"
max-pages-per-part = 50  # Automatically split the document for translation and merge it back.
only_include_translated_page = false # Only include translated pages in the output PDF. Effective only when `pages` is used.
# no-watermark = false  # DEPRECATED: Use watermark-output-mode instead
skip-scanned-detection = false  # Skip scanned document detection for faster processing
auto_extract_glossary = true # Set to false to disable automatic term extraction
formular_font_pattern = "" # Font pattern for formula text
formular_char_pattern = "" # Character pattern for formula text
show_char_box = false # Show character bounding boxes (debug)
ocr_workaround = false # Use OCR workaround for scanned PDFs
rpc_doclayout = "" # RPC service host for document layout analysis
working_dir = "" # Working directory for translation
auto_enable_ocr_workaround = false # Enable automatic OCR workaround for scanned PDFs. See docs for interaction with ocr_workaround and skip_scanned_detection.
skip_form_render = false # Skip form rendering (default: False)
skip_curve_render = false # Skip curve rendering (default: False)
only_parse_generate_pdf = false # Only parse PDF and generate output PDF without translation (default: False)
remove_non_formula_lines = false # Remove non-formula lines from paragraph areas (default: False)
non_formula_line_iou_threshold = 0.2 # IoU threshold for paragraph overlap detection (default: 0.2)
figure_table_protection_threshold = 0.3 # IoU threshold for figure/table protection (default: 0.3)

# Translation service
openai = true
openai-model = "gpt-4o-mini"
openai-base-url = "https://api.openai.com/v1"
openai-api-key = "your-api-key-here"
pool-max-workers = 8  # Maximum worker threads for task processing (defaults to QPS value if not set)

# Glossary Options (Optional)
# glossary-files = "/path/to/glossary1.csv,/path/to/glossary2.csv"

# Output control
no-dual = false
no-mono = false
min-text-length = 5
report-interval = 0.5

# Offline assets management
# Uncomment one of these options as needed:
# generate-offline-assets = "/path/to/output/dir"
# restore-offline-assets = "/path/to/offline_assets_package.zip"
```

## Python API

The recommended way to use BabelDOC in Python is to call `high_level.do_translate_async_stream` from [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

>   **All APIs of BabelDOC should be considered as internal APIs, and any direct use of BabelDOC is not supported.**

## Preview

<div align="center">
    <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## We are hiring

See details: [EN](https://github.com/funstory-ai/jobs) | [ZH](https://github.com/funstory-ai/jobs/blob/main/README_ZH.md)

## Background

BabelDOC aims to provide a standard pipeline and interface for PDF parsing and translation.

## Roadmap

*   Add line support
*   Add table support
*   Add cross-page/cross-column paragraph support
*   More advanced typesetting features
*   Outline support
*   ...

## Versioning

This project uses Semantic Versioning with Pride Versioning: "0.MAJOR.MINOR".

## Known Issues

1.  Parsing errors in author and reference sections.
2.  Line support is not yet implemented.
3.  Drop caps are not supported.
4.  Large pages may be skipped.

## How to Contribute

Contributions are welcome!  See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

## Acknowledgements

*   [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
*   [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
*   [pdfminer](https://github.com/pdfminer/pdfminer.six)
*   [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
*   [Asynchronize](https://github.com/multimeric/Asynchronize/tree/master?tab=readme-ov-file)
*   [PriorityThreadPoolExecutor](https://github.com/oleglpts/PriorityThreadPoolExecutor)

##  Star History

<a href="https://star-history.com/#funstory-ai/babeldoc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date"/>
 </picture>
</a>

>   **Important Interaction Note for `--auto-enable-ocr-workaround`:**
>
>   When `--auto-enable-ocr-workaround` is set to `true`:
>
>   1.  `ocr_workaround` and `skip_scanned_detection` are forced to `false` by `TranslationConfig`.
>   2.  If the document is heavily scanned and `auto_enable_ocr_workaround` is `true`, the system attempts to set both `ocr_workaround` to `true` and `skip_scanned_detection` to `true`.