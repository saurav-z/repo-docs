# BabelDOC: Effortlessly Translate PDF Documents

**Translate scientific papers and PDF documents with ease!** BabelDOC is a powerful library for translating PDF documents, providing both online service and self-deployment options.  [Explore BabelDOC on GitHub](https://github.com/funstory-ai/BabelDOC).

[![PyPI Version](https://img.shields.io/pypi/v/BabelDOC)](https://pypi.org/project/BabelDOC/)
[![PyPI Downloads](https://static.pepy.tech/badge/BabelDOC)](https://pepy.tech/projects/BabelDOC)
[![GitHub License](https://img.shields.io/github/license/funstory-ai/BabelDOC)](./LICENSE)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white)](https://t.me/+Z9_SgnxmsmA5NzBl)

<a href="https://trendshift.io/repositories/13358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

## Key Features

*   **PDF Translation:** Accurately translates PDF documents, including scientific papers.
*   **Online Service:** Try the beta version with [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) (1000 free pages/month).
*   **Self-Deployment:**  Deploy your own translation service with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).
*   **Command-Line Interface:**  Provides a simple CLI for direct use (primarily for debugging).
*   **Python API:** Integrates seamlessly into your Python projects.
*   **Bilingual Output:** Generate translated PDFs with original and translated text side-by-side.
*   **Offline Assets:**  Supports offline asset management for environments without internet access.

## How to Get Started

### 1. Install (Recommended)

We recommend using [uv](https://github.com/astral-sh/uv) for package management:

1.  **Install `uv`:** Follow the [uv installation instructions](https://github.com/astral-sh/uv#installation) and set up your `PATH` environment variable.
2.  **Install BabelDOC:**

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

3.  **Run BabelDOC:**

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    ```

### 2. Install from Source (Alternative)

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    ```

2.  **Install Dependencies:**

    ```bash
    uv run babeldoc --help
    ```

3.  **Run BabelDOC:**

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

## Preview

<div align="center">
<img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## Advanced Options

**Note:**  The CLI is primarily for debugging and advanced users.  For ease of use, use the [online service](https://app.immersivetranslate.com/babel-doc/) or self-deploy with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

### Language Options

*   `--lang-in`, `-li`: Source language code (default: `en`)
*   `--lang-out`, `-lo`: Target language code (default: `zh`)

### PDF Processing Options

*   `--files`: PDF file paths.
*   `--pages`, `-p`: Pages to translate (e.g., "1,2,1-,-3,3-5"). Defaults to all pages.
*   `--split-short-lines`:  Force split short lines.
*   `--short-line-split-factor`:  Split threshold factor (default: 0.8).
*   `--skip-clean`: Skip PDF cleaning.
*   `--dual-translate-first`: Place translated pages first.
*   `--disable-rich-text-translate`: Disable rich text translation.
*   `--enhance-compatibility`: Enable all compatibility options.
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF.
*   `--watermark-output-mode`: Control watermark output mode: 'watermarked', 'no_watermark', 'both'.
*   `--max-pages-per-part`: Maximum pages per split part.
*   `--translate-table-text`: Translate table text (experimental, default: False)
*   `--formular-font-pattern`: Font pattern to identify formula text (default: None)
*   `--formular-char-pattern`: Character pattern to identify formula text (default: None)
*   `--show-char-box`: Show character bounding boxes (debug only, default: False)
*   `--skip-scanned-detection`: Skip scanned document detection (default: False)
*   `--ocr-workaround`: Use OCR workaround (default: False).
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False).
*   `--primary-font-family`: Override primary font family.
*   `--only-include-translated-page`: Only include translated pages in the output PDF.
*   `--merge-alternating-line-numbers`: Enable post-processing to merge alternating line-number layouts (off by default).
*   `--skip-form-render`: Skip form rendering (default: False).
*   `--skip-curve-render`: Skip curve rendering (default: False).
*   `--only-parse-generate-pdf`: Only parse PDF and generate output PDF without translation (default: False).
*   `--remove-non-formula-lines`: Remove non-formula lines (default: False).
*   `--non-formula-line-iou-threshold`: IoU threshold for removing non-formula lines (default: 0.9).
*   `--figure-table-protection-threshold`: IoU threshold for protecting lines in figure/table areas when removing non-formula lines (default: 0.9).

*   `--rpc-doclayout`: RPC service host address for document layout analysis (default: None)
*   `--working-dir`: Working directory for translation. If not set, use temp directory.
*   `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file. If not set, the glossary will not be saved.

### Translation Service Options

*   `--qps`: Queries Per Second limit (default: 4).
*   `--ignore-cache`: Force retranslation.
*   `--no-dual`: Do not output bilingual PDFs.
*   `--no-mono`: Do not output monolingual PDFs.
*   `--min-text-length`: Minimum text length to translate (default: 5).
*   `--openai`: Use OpenAI for translation (default: False).
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation. (Currently not recommended, it may affect translation quality, default: False)
*   `--pool-max-workers`: Maximum worker threads for internal task processing pools. If not specified, defaults to QPS value. This parameter directly sets the worker count, replacing previous QPS-based dynamic calculations.
*   `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model (default: `gpt-4o-mini`).
*   `--openai-base-url`: OpenAI API base URL.
*   `--openai-api-key`: OpenAI API key.

### Glossary Options

*   `--glossary-files`: Paths to glossary CSV files. CSV files must have 'source', 'target', and optional 'tgt_lng' columns.

### Output Control

*   `--output`, `-o`: Output directory.
*   `--debug`: Enable debug logging.
*   `--report-interval`: Progress report interval (default: 0.1 seconds).

### General Options

*   `--warmup`: Only download and verify assets then exit.

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package.
*   `--restore-offline-assets`: Restore an offline assets package.

## Configuration File

Use TOML format for configuration files.

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

For programmatic access, call the `high_level.do_translate_async_stream` function of [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

> [!WARNING]
> **All APIs of BabelDOC should be considered as internal APIs, and any direct use of BabelDOC is not supported.**

## Background & Related Projects

BabelDOC builds upon established techniques in PDF parsing and translation, drawing inspiration from projects like:

*   [mathpix](https://mathpix.com/)
*   [Doc2X](https://doc2x.noedgeai.com/)
*   [minerU](https://github.com/opendatalab/MinerU)
*   [PDFMathTranslate](https://github.com/funstory-ai/yadt)

## Roadmap

Planned improvements:

*   Line Support
*   Table Support
*   Cross-page/column paragraph support
*   Advanced typesetting features
*   Outline support

Our 1.0 goal is to finish a translation from [PDF Reference, Version 1.7](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.7old.pdf) to Simplified Chinese, Traditional Chinese, Japanese, and Spanish.

## Versioning

This project uses Semantic Versioning combined with Pride Versioning. Format: "0.MAJOR.MINOR".

## Known Issues

1.  Parsing errors in author/reference sections.
2.  Line support is limited.
3.  Drop caps not supported.
4.  Large pages may be skipped.

## How to Contribute

We welcome contributions!  See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

Follow the YADT [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).

[Immersive Translation](https://immersivetranslate.com) sponsors monthly Pro membership redemption codes for active contributors. See [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md) for details.

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