---
title: BabelDOC: Scientific PDF Translation and Bilingual Comparison
description: Translate scientific PDF papers effortlessly with BabelDOC, featuring online and self-hosted options.  Utilize a simple command-line interface and powerful Python API to unlock multilingual research.
keywords: PDF translation, scientific paper translation, bilingual comparison, Python API, command-line tool, OpenAI, Immersive Translate
---

# BabelDOC: Translate Scientific PDFs with Ease

BabelDOC is your go-to solution for translating and comparing scientific PDF documents. Whether you need a quick translation or a deep dive into bilingual analysis, BabelDOC provides the tools you need.  **Unlock research insights faster – get started today!** ([View the original repo](https://github.com/funstory-ai/BabelDOC))

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
  <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
</picture>

[![PyPI](https://img.shields.io/pypi/v/BabelDOC)](https://pypi.org/project/BabelDOC/)
[![Downloads](https://static.pepy.tech/badge/BabelDOC)](https://pepy.tech/projects/BabelDOC)
[![License](https://img.shields.io/github/license/funstory-ai/BabelDOC)](./LICENSE)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white)](https://t.me/+Z9_SgnxmsmA5NzBl)

<a href="https://trendshift.io/repositories/13358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

## Key Features

*   **Online Service:** Try the beta version with [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) for 1000 free pages per month.
*   **Self-Deployment:** Integrate BabelDOC with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for a web UI and access to more translation services.
*   **Command-Line Interface (CLI):** Quickly translate PDFs with a simple command-line tool.
*   **Python API:** Integrate BabelDOC into your Python projects for automated PDF processing.
*   **Bilingual Comparison:** Easily compare original and translated documents side-by-side or in alternating pages.
*   **OpenAI Integration:** Leverage the power of OpenAI for high-quality translations.
*   **Glossary Support:**  Improve translation accuracy by providing custom glossaries.
*   **Offline Assets Management:** Download and manage all assets locally for environments without internet.

## Getting Started

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) for installation and dependency management.

1.  **Install uv:** Follow the instructions at [uv installation](https://github.com/astral-sh/uv#installation).
2.  **Install BabelDOC:**

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

3.  **Use the CLI:**

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf

    # Multiple files
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```

### Install from Source

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    ```

2.  **Run babeldoc:**

    ```bash
    uv run babeldoc --help

    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"

    # Multiple files
    uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

## Advanced Options

**Note:**  The CLI is primarily for debugging.  For end-users, we recommend the [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) online service or self-deployment using [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

### Language Options

*   `--lang-in`, `-li`: Source language code (default: `en`)
*   `--lang-out`, `-lo`: Target language code (default: `zh`)

    > **Tip:** Currently optimized for English-to-Chinese translation.

### PDF Processing Options

*   `--files`: Input PDF file(s).
*   `--pages`, `-p`: Pages to translate (e.g., "1,2,1-,-3,3-5").
*   `--split-short-lines`: Force split short lines into different paragraphs.
*   `--short-line-split-factor`: Split threshold factor (default: 0.8).
*   `--skip-clean`: Skip PDF cleaning step.
*   `--dual-translate-first`: Put translated pages first in dual PDF mode (default: original pages first).
*   `--disable-rich-text-translate`: Disable rich text translation.
*   `--enhance-compatibility`: Enable all compatibility enhancement options (equivalent to `--skip-clean --dual-translate-first --disable-rich-text-translate`).
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF.
*   `--watermark-output-mode`: Control watermark output mode: 'watermarked' (default), 'no_watermark', or 'both'.
*   `--max-pages-per-part`: Maximum pages per split translation part.
*   `--translate-table-text`: Translate table text (experimental, default: False).
*   `--formular-font-pattern`: Font pattern to identify formula text.
*   `--formular-char-pattern`: Character pattern to identify formula text.
*   `--show-char-box`: Show character bounding boxes (debug only, default: False).
*   `--skip-scanned-detection`: Skip scanned document detection (default: False).
*   `--ocr-workaround`: Use OCR workaround (default: False).
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False). See "Important Interaction Note" below.
*   `--primary-font-family`: Override primary font family.
*   `--only-include-translated-page`: Only include translated pages in the output PDF (when `--pages` is used, default: False).
*   `--merge-alternating-line-numbers`: Enable post-processing to merge alternating line-number layouts (off by default).
*   `--skip-form-render`: Skip form rendering (default: False).
*   `--skip-curve-render`: Skip curve rendering (default: False).
*   `--only-parse-generate-pdf`: Only parse PDF and generate output PDF without translation (default: False).
*   `--remove-non-formula-lines`: Remove non-formula lines from paragraph areas (default: False).
*   `--non-formula-line-iou-threshold`: IoU threshold for detecting paragraph overlap when removing non-formula lines (default: 0.9).
*   `--figure-table-protection-threshold`: IoU threshold for protecting lines in figure/table areas (default: 0.9).
*   `--rpc-doclayout`: RPC service host address for document layout analysis (default: None).
*   `--working-dir`: Working directory for translation.
*   `--no-auto-extract-glossary`: Disable automatic term extraction (defaults to enabled).
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file.

    > **Tips:** Use `--enhance-compatibility` for PDF compatibility. Use `--max-pages-per-part` for large documents. Use `--skip-scanned-detection` if the document is not scanned. Use `--ocr-workaround` for scanned PDFs.

### Translation Service Options

*   `--qps`: Queries Per Second limit for translation service (default: 4).
*   `--ignore-cache`: Ignore translation cache.
*   `--no-dual`: Do not output bilingual PDF files.
*   `--no-mono`: Do not output monolingual PDF files.
*   `--min-text-length`: Minimum text length to translate (default: 5).
*   `--openai`: Use OpenAI for translation (default: False).
*   `--custom-system-prompt`: Custom system prompt.
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation.
*   `--pool-max-workers`: Maximum worker threads for internal task processing pools.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.

    > **Tips:** Currently, only OpenAI-compatible LLMs are supported.  Use `--custom-system-prompt` to add `/no_think` instruction of Qwen 3 in the prompt.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model (default: gpt-4o-mini).
*   `--openai-base-url`: OpenAI API base URL.
*   `--openai-api-key`: OpenAI API key.

    > **Tips:** Use any OpenAI-compatible API endpoint. For local models, any API key can be used.

### Glossary Options

*   `--glossary-files`: Comma-separated paths to glossary CSV files.

    *   Each CSV should have `source`, `target`, and (optional) `tgt_lng` columns.
    *   The `source` is the original term.
    *   The `target` is the translated term.
    *   `tgt_lng` specifies the target language (e.g., "zh-CN"). If omitted, the glossary entry is applicable for all target languages.

    The glossary is included in the prompt to the LLM, along with an instruction to adhere to it.

### Output Control

*   `--output`, `-o`: Output directory.
*   `--debug`: Enable debug logging and export intermediate results.
*   `--report-interval`: Progress report interval in seconds (default: 0.1).

### General Options

*   `--warmup`: Only download and verify assets, then exit.

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package.
*   `--restore-offline-assets`: Restore an offline assets package.

    > **Tips:**  Use for offline environments. Generate the package on a machine with internet access first. Verify SHA3-256 hashes.

### Configuration File

*   `--config`, `-c`: Configuration file path (TOML format).

```toml
# Example Configuration
[babeldoc]
debug = true
lang-in = "en-US"
lang-out = "zh-CN"
qps = 10
output = "/path/to/output/dir"
openai = true
openai-model = "gpt-4o-mini"
openai-base-url = "https://api.openai.com/v1"
openai-api-key = "your-api-key-here"
```

## Python API

The recommended way to call BabelDOC in Python is to call the `high_level.do_translate_async_stream` function of [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

> **Warning:** Direct use of BabelDOC APIs is not supported.

## Background

BabelDOC builds upon and aims to improve existing document processing and translation solutions. This project focuses on creating a standardized pipeline with plugin architecture.

## Roadmap

*   [ ] Add line support
*   [ ] Add table support
*   [ ] Add cross-page/cross-column paragraph support
*   [ ] More advanced typesetting features
*   [ ] Outline support
*   [ ] ...

The initial 1.0 version will target the translation of the [PDF Reference, Version 1.7](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.7old.pdf) into Simplified Chinese, Traditional Chinese, Japanese, and Spanish.

## Versioning

This project uses [Semantic Versioning](https://semver.org/) combined with [Pride Versioning](https://pridever.org/). The format is: "0.MAJOR.MINOR".

> **Note:** API compatibility refers to the compatibility with [pdf2zh_next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

## Known Issues

1.  Parsing errors in author/reference sections.
2.  Line support missing.
3.  Drop caps not supported.
4.  Large pages skipped.

## How to Contribute

Contribute to BabelDOC! See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide. Follow the [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).

Active contributors receive monthly Pro membership redemption codes for [Immersive Translation](https://immersivetranslate.com) - see [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md).

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
> 1.  `auto_enable_ocr_workaround` will force `ocr_workaround` and `skip_scanned_detection` to `false` during the initial setup.
> 2.  If a scanned document is detected and `auto_enable_ocr_workaround` is `true`, the system will attempt to set `ocr_workaround` and `skip_scanned_detection` to `true`.  This effectively gives the system control over OCR processing for scanned documents.