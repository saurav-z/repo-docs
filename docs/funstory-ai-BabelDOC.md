---
title: BabelDOC: Effortlessly Translate PDF Documents with AI-Powered Precision
description: BabelDOC is a powerful Python library for translating PDF documents.  It offers online and self-hosted options, a command-line interface, and a Python API.  Translate scientific papers, technical documents, and more with ease.
keywords: PDF translation, document translation, AI translation, Python library, BabelDOC, scientific paper translation, bilingual PDF, OpenAI, PDFMathTranslate, Immersive Translate
---

<div align="center">
    <a href="https://github.com/funstory-ai/BabelDOC">
        <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC" />
    </a>
    <br/>
    <p>
        <a href="https://pypi.org/project/BabelDOC/">
            <img src="https://img.shields.io/pypi/v/BabelDOC" alt="PyPI Version" />
        </a>
        <a href="https://pepy.tech/projects/BabelDOC">
            <img src="https://static.pepy.tech/badge/BabelDOC" alt="Downloads" />
        </a>
        <a href="./LICENSE">
            <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC" alt="License" />
        </a>
        <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
            <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white" alt="Telegram" />
        </a>
    </p>
    <a href="https://trendshift.io/repositories/13358" target="_blank">
        <img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55" />
    </a>
</div>

## BabelDOC: Translate PDF Documents with AI-Powered Precision

**BabelDOC is your go-to solution for seamlessly translating PDF documents, providing both online and self-deployment options.** This open-source library empowers you to translate scientific papers, technical documents, and more, making information accessible in multiple languages.

*   **Versatile Translation:** Translate documents from English to Chinese, and beyond.
*   **Online & Self-Hosted Options:** Utilize the beta online service ([Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/)) or self-deploy with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).
*   **Flexible Interface:** Command-line interface (CLI) and Python API for easy integration into your workflow.
*   **Dual-Language Support:** Create bilingual PDFs for side-by-side comparison and learning.
*   **OpenAI Integration:** Leverage the power of OpenAI models for high-quality translations.
*   **Configurable:** Offers a wide array of options to customize the translation process, including glossary support.

[View Supported Languages](https://funstory-ai.github.io/BabelDOC/supported_languages/)

## Key Features

*   **AI-Powered Translation:** Utilizes state-of-the-art AI for accurate and natural-sounding translations.
*   **Bilingual PDF Generation:** Generates side-by-side bilingual PDFs for easy comparison and study.
*   **Command-Line Interface:** Simple and intuitive CLI for quick translation tasks.
*   **Python API:** Integrate BabelDOC into your Python projects for automated translation workflows.
*   **Online Service Integration:** Easy access to a user-friendly online interface ([Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/)).
*   **Self-Deployment Options:** Flexibility to self-host the translation service using [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).
*   **Glossary Support:** Incorporate custom glossaries to ensure terminology consistency.
*   **Offline Assets Management:** Package and restore offline assets for use in environments without internet access.

## Preview

<div align="center">
<img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%" alt="BabelDOC Preview"/>
</div>

## Getting Started

### Install Using uv (Recommended)

1.  **Install uv:** Refer to the [uv installation guide](https://github.com/astral-sh/uv#installation) and set up the `PATH` environment variable.

2.  **Install BabelDOC:**

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

3.  **Use the `babeldoc` command:**

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here" --files example.pdf
    ```

    For multiple files:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here" --files example1.pdf --files example2.pdf
    ```

### Install from Source (with uv)

1.  **Install uv:** (See above)

2.  **Clone the repository and install dependencies:**

    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    uv run babeldoc --help
    ```

3.  **Use the `uv run babeldoc` command:**

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

    For multiple files:

    ```bash
    uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

## Advanced Options

> **Note:**  The CLI is primarily for debugging.  For end-users, the recommended options are the **Online Service** ([Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/)) or self-hosting with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

### Language Options

*   `--lang-in`, `-li`: Source language code (default: `en`)
*   `--lang-out`, `-lo`: Target language code (default: `zh`)

### PDF Processing Options

*   `--files`: Input PDF files.
*   `--pages`, `-p`: Pages to translate (e.g., "1,2,1-,-3,3-5"). If not set, all pages are translated.
*   `--split-short-lines`: Force split short lines.
*   `--short-line-split-factor`: Split threshold factor (default: 0.8).
*   `--skip-clean`: Skip PDF cleaning.
*   `--dual-translate-first`: Put translated pages first in dual PDF mode (default: original pages first).
*   `--disable-rich-text-translate`: Disable rich text translation.
*   `--enhance-compatibility`: Enable all compatibility enhancement options.
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF.
*   `--watermark-output-mode`: Control watermark output mode: 'watermarked' (default), 'no_watermark', 'both'.
*   `--max-pages-per-part`: Maximum pages per split translation part.
*   `--translate-table-text`: Translate table text (experimental, default: False)
*   `--formular-font-pattern`: Font pattern for formula text (default: None)
*   `--formular-char-pattern`: Character pattern for formula text (default: None)
*   `--show-char-box`: Show character bounding boxes (debug only, default: False)
*   `--skip-scanned-detection`: Skip scanned document detection (default: False).
*   `--ocr-workaround`: Use OCR workaround (default: False).
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False).
*   `--primary-font-family`: Override primary font family.
*   `--only-include-translated-page`: Only include translated pages in the output PDF (when `--pages` is used).
*   `--merge-alternating-line-numbers`: Merge alternating line-number layouts.
*   `--skip-form-render`: Skip form rendering (default: False).
*   `--skip-curve-render`: Skip curve rendering (default: False).
*   `--only-parse-generate-pdf`: Only parse PDF and generate output PDF without translation (default: False).
*   `--remove-non-formula-lines`: Remove non-formula lines from paragraph areas (default: False).
*   `--non-formula-line-iou-threshold`: IoU threshold for detecting paragraph overlap when removing non-formula lines (default: 0.9).
*   `--figure-table-protection-threshold`: IoU threshold for protecting lines in figure/table areas when removing non-formula lines (default: 0.9).

*   `--rpc-doclayout`: RPC service host for document layout analysis (default: None)
*   `--working-dir`: Working directory.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file.

### Translation Service Options

*   `--qps`: QPS (Queries Per Second) limit (default: 4)
*   `--ignore-cache`: Ignore translation cache.
*   `--no-dual`: Do not output bilingual PDF files.
*   `--no-mono`: Do not output monolingual PDF files.
*   `--min-text-length`: Minimum text length to translate (default: 5)
*   `--openai`: Use OpenAI for translation (default: False)
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint.
*   `--pool-max-workers`: Maximum worker threads.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model (default: gpt-4o-mini)
*   `--openai-base-url`: OpenAI API base URL.
*   `--openai-api-key`: OpenAI API key.

### Glossary Options

*   `--glossary-files`: Comma-separated paths to glossary CSV files.

### Output Control

*   `--output`, `-o`: Output directory.
*   `--debug`: Enable debug logging.
*   `--report-interval`: Progress report interval (default: 0.1).

### General Options

*   `--warmup`: Only download and verify required assets then exit (default: False)

### Offline Assets Management

*   `--generate-offline-assets`: Generate offline assets package.
*   `--restore-offline-assets`: Restore offline assets package.

### Configuration File

*   `--config`, `-c`: Configuration file path (TOML format).

Example Configuration (TOML):

```toml
[babeldoc]
debug = true
lang-in = "en-US"
lang-out = "zh-CN"
qps = 10
output = "/path/to/output/dir"
split-short-lines = false
short-line-split-factor = 0.8
skip-clean = false
dual-translate-first = false
disable-rich-text-translate = false
use-alternating-pages-dual = false
watermark-output-mode = "watermarked"
max-pages-per-part = 50
only_include_translated_page = false
skip-scanned-detection = false
auto_extract_glossary = true
formular_font_pattern = ""
formular_char_pattern = ""
show-char-box = false
ocr_workaround = false
rpc_doclayout = ""
working_dir = ""
auto_enable_ocr_workaround = false
skip_form_render = false
skip_curve_render = false
only_parse_generate_pdf = false
remove_non_formula_lines = false
non_formula_line_iou_threshold = 0.2
figure_table_protection_threshold = 0.3
openai = true
openai-model = "gpt-4o-mini"
openai-base-url = "https://api.openai.com/v1"
openai-api-key = "your-api-key-here"
pool-max-workers = 8
no-dual = false
no-mono = false
min-text-length = 5
report-interval = 0.5
# generate-offline-assets = "/path/to/output/dir"
# restore-offline-assets = "/path/to/offline_assets_package.zip"
```

## Python API

**Note:** BabelDOC's internal APIs are subject to change. It is recommended to call the `high_level.do_translate_async_stream` function of [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for Python API usage.

## Background

[Original Repo](https://github.com/funstory-ai/BabelDOC)

BabelDOC builds upon and extends previous projects like [PDFMathTranslate](https://github.com/funstory-ai/yadt) and leverages technologies like [layoutreader](https://github.com/microsoft/unilm/tree/master/layoutreader) for PDF parsing and translation. It aims to provide a standard pipeline and interface for efficient document translation.

## Roadmap

*   Add line support
*   Add table support
*   Add cross-page/cross-column paragraph support
*   More advanced typesetting features
*   Outline support

## Version Numbering

Uses Semantic Versioning and Pride Versioning.  Format: "0.MAJOR.MINOR".

## Known Issues

1.  Parsing errors in author and reference sections.
2.  Lines are not supported.
3.  Drop caps are not supported.
4.  Large pages may be skipped.

## How to Contribute

Contribute to BabelDOC!  See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide for details.

The [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md) is in effect.

Active contributors can receive monthly Pro membership redemption codes from [Immersive Translation](https://immersivetranslate.com) as rewards ([CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md)).

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

> **Important Interaction Note for `--auto-enable-ocr-workaround`:**
>
> 1.  `ocr_workaround` and `skip_scanned_detection` are initially set to `false` when `--auto-enable-ocr-workaround` is set.
> 2.  If a document is heavily scanned and `auto_enable_ocr_workaround` is true, the system will attempt to set both `ocr_workaround` to `true` and `skip_scanned_detection` to `true`.