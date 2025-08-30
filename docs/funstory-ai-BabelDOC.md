---
title: BabelDOC: Your AI-Powered PDF Translation Solution
description: Translate scientific papers and documents effortlessly with BabelDOC, a Python library offering online and self-hosted options. Enhance your research and understanding with accurate translations.
keywords: PDF translation, scientific paper translation, document translator, AI translation, Python, BabelDOC, Immersive Translate, PDFMathTranslate
---

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

## BabelDOC: Effortlessly Translate PDFs with AI 

**BabelDOC is a powerful Python library designed to translate PDF documents, perfect for scientific papers and other complex documents.**  Offering both online and self-hosted options, BabelDOC empowers you to break down language barriers and access information in your preferred language. [Visit the original repository](https://github.com/funstory-ai/BabelDOC).

## Key Features

*   **Online Service:**  Try the beta version at [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/), offering 1000 free pages per month.
*   **Self-Deployment:** Deploy your own instance using [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next), including a WebUI with various translation services.
*   **Command-Line Interface (CLI):**  Translate documents directly from your terminal.
*   **Python API:** Integrate BabelDOC into your own Python projects.
*   **Bilingual Output:**  Generate translated PDFs alongside the original documents for easy comparison.
*   **Advanced PDF Processing:**  Includes options for page selection, splitting, cleaning, and handling scanned documents.
*   **Glossary Support:** Improve accuracy by providing custom glossaries for specific terminology.
*   **Offline Asset Management:** Create and restore offline packages for environments without internet access.
*   **OpenAI Integration:** Leverage the power of OpenAI for high-quality translations.

## Getting Started

### Install

We recommend using [uv](https://github.com/astral-sh/uv) to manage your virtual environment.

#### 1. Install `uv`

Follow the [uv installation instructions](https://github.com/astral-sh/uv#installation) and set up the `PATH` environment variable.

#### 2. Install BabelDOC

**From PyPI:**

```bash
uv tool install --python 3.12 BabelDOC
babeldoc --help
```

**From Source:**

```bash
git clone https://github.com/funstory-ai/BabelDOC
cd BabelDOC
uv run babeldoc --help
```

#### 3. Run the `babeldoc` command

**From PyPI:**

```bash
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
```

**From Source:**

```bash
uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
```

## Advanced Options (CLI)

> **Note:** The CLI is primarily for debugging. For end-users, use the [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) online service (1000 free pages/month) or the self-deployable [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

This section covers the available CLI options for customizing your translation process.

### Language Options

*   `--lang-in`, `-li`: Source language code (default: `en`)
*   `--lang-out`, `-lo`: Target language code (default: `zh`)

### PDF Processing Options

*   `--files`: Input PDF document paths.
*   `--pages`, `-p`: Pages to translate (e.g., "1,2,1-,-3,3-5").  If not set, all pages are translated.
*   `--split-short-lines`: Split short lines into paragraphs (may cause issues).
*   `--short-line-split-factor`: Split threshold (default: 0.8).
*   `--skip-clean`: Skip PDF cleaning.
*   `--dual-translate-first`: Translated pages first in dual PDF (default: original first).
*   `--disable-rich-text-translate`: Disable rich text translation.
*   `--enhance-compatibility`: Enable all compatibility options.
*   `--use-alternating-pages-dual`: Use alternating pages for dual PDF output.
*   `--watermark-output-mode`: Control watermark mode:  `watermarked` (default), `no_watermark`, or `both`.
*   `--max-pages-per-part`: Maximum pages per split translation part.
*   `--translate-table-text`: Translate table text (experimental, default: `False`).
*   `--formular-font-pattern`: Font pattern for formulas.
*   `--formular-char-pattern`: Character pattern for formulas.
*   `--show-char-box`: Show character bounding boxes (debug).
*   `--skip-scanned-detection`: Skip scanned document detection.
*   `--ocr-workaround`: Apply OCR workaround (default: `False`).
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: `False`).
*   `--primary-font-family`: Override font family (serif, sans-serif, script).
*   `--only-include-translated-page`: Include translated pages only (requires `--pages`).
*   `--merge-alternating-line-numbers`: Merge alternating line numbers.
*   `--skip-form-render`: Skip form rendering.
*   `--skip-curve-render`: Skip curve rendering.
*   `--only-parse-generate-pdf`: Only parse PDF and generate output PDF without translation.
*   `--remove-non-formula-lines`: Remove non-formula lines from paragraph areas.
*   `--non-formula-line-iou-threshold`: IoU threshold for detecting paragraph overlap when removing non-formula lines (default: 0.9).
*   `--figure-table-protection-threshold`: IoU threshold for protecting lines in figure/table areas when removing non-formula lines (default: 0.9).
*   `--rpc-doclayout`: RPC service host for document layout analysis.
*   `--working-dir`: Working directory for translation.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.
*   `--save-auto-extracted-glossary`: Save the extracted glossary.

### Translation Service Options

*   `--qps`: Queries per second (default: 4).
*   `--ignore-cache`: Force retranslation.
*   `--no-dual`: Don't output bilingual PDFs.
*   `--no-mono`: Don't output monolingual PDFs.
*   `--min-text-length`: Minimum text length to translate (default: 5).
*   `--openai`: Use OpenAI for translation (default: `False`).
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hints (discouraged).
*   `--pool-max-workers`: Worker threads for internal task processing.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model (default: `gpt-4o-mini`).
*   `--openai-base-url`: OpenAI API base URL.
*   `--openai-api-key`: OpenAI API key.

### Glossary Options

*   `--glossary-files`: Comma-separated paths to glossary CSV files.  Each CSV should include `source`, `target`, and optionally `tgt_lng` columns.

### Output Control

*   `--output`, `-o`: Output directory.
*   `--debug`: Enable debug logging.
*   `--report-interval`: Progress report interval (default: 0.1).

### General Options

*   `--warmup`: Download and verify assets then exit.

### Offline Assets Management

*   `--generate-offline-assets`: Create an offline assets package.
*   `--restore-offline-assets`: Restore an offline assets package.

## Python API

The recommended way to call BabelDOC in Python is to use `high_level.do_translate_async_stream` function from [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

## Roadmap

*   Add line support
*   Add table support
*   Add cross-page/cross-column paragraph support
*   More advanced typesetting features
*   Outline support
*   ...

## Versioning

BabelDOC uses a combination of [Semantic Versioning](https://semver.org/) and [Pride Versioning](https://pridever.org/) with the format "0.MAJOR.MINOR".

## Known Issues

1.  Parsing errors in the author and reference sections.
2.  Lines are not supported.
3.  Does not support drop caps.
4.  Large pages will be skipped.

## How to Contribute

Contributions are welcome! See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

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