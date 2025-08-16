<!-- # BabelDOC: PDF Translation Library -->

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

## BabelDOC: Translate PDF Documents with Ease

BabelDOC is a powerful Python library designed to translate scientific papers and other PDF documents, offering both online and self-hosted solutions.  Explore the [original repo](https://github.com/funstory-ai/BabelDOC) for more details.

**Key Features:**

*   **Online Translation Service:**  Utilize the beta version of [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) for free translation with 1000 pages per month.
*   **Self-Deployment:** Deploy your own translation service with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).
*   **Command Line Interface (CLI):**  Translate PDFs directly from your terminal.
*   **Python API:** Integrate BabelDOC into your own Python applications.
*   **Dual-Language PDF Output:**  Generate side-by-side or alternating page bilingual PDFs.
*   **Advanced Options:**  Fine-tune translation with language, PDF processing, translation service, and output control options.
*   **Offline Assets:** Generate and restore offline asset packages for environments without internet access.

##  Getting Started

BabelDOC can be installed via PyPI or from source.  We strongly recommend using [uv](https://github.com/astral-sh/uv) for environment management.

### Install Using `uv` and PyPI

1.  Install `uv` and configure your `PATH` environment variable.
2.  Install BabelDOC:

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```
3.  Use the `babeldoc` command:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    ```
    or for multiple files
      ```bash
      babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
      ```

### Install from Source

1.  Install `uv` and configure your `PATH` environment variable.
2.  Clone the repository and install dependencies:

    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    uv run babeldoc --help
    ```

3.  Run BabelDOC:

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```
    or for multiple files
    ```bash
    uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

##  Advanced Options

### Language Options

*   `--lang-in`, `-li`: Source language code (default: en)
*   `--lang-out`, `-lo`: Target language code (default: zh)

### PDF Processing Options

*   `--files`: Input PDF document paths.
*   `--pages`, `-p`: Specify pages to translate.
*   `--split-short-lines`:  Split short lines into paragraphs.
*   `--short-line-split-factor`: Split threshold factor (default: 0.8).
*   `--skip-clean`: Skip PDF cleaning step.
*   `--dual-translate-first`: Put translated pages first in dual PDF mode (default: original pages first).
*   `--disable-rich-text-translate`: Disable rich text translation.
*   `--enhance-compatibility`: Enable all compatibility enhancement options.
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF.
*   `--watermark-output-mode`: Control watermark output mode: 'watermarked' (default) adds watermark, 'no_watermark' doesn't add watermark, 'both' outputs both versions.
*   `--max-pages-per-part`:  Split large documents.
*   `--translate-table-text`: Translate table text (experimental).
*   `--formular-font-pattern`: Font pattern for formula text.
*   `--formular-char-pattern`: Character pattern for formula text.
*   `--show-char-box`: Show character bounding boxes (debug).
*   `--skip-scanned-detection`: Skip scanned document detection.
*   `--ocr-workaround`: Use OCR workaround.
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (see "Important Interaction Note" below).
*   `--primary-font-family`: Override primary font family.
*   `--only-include-translated-page`: Only include translated pages in the output PDF.
*   `--rpc-doclayout`: RPC service host address for document layout analysis.
*   `--working-dir`: Working directory for translation.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary to a file.

### Translation Service Options

*   `--qps`: Queries Per Second limit (default: 4).
*   `--ignore-cache`: Ignore translation cache.
*   `--no-dual`: Do not output bilingual PDF files.
*   `--no-mono`: Do not output monolingual PDF files.
*   `--min-text-length`: Minimum text length to translate (default: 5).
*   `--openai`: Use OpenAI for translation (default: False).
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation. (Currently not recommended).
*   `--pool-max-workers`: Maximum worker threads for internal task processing pools.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model (default: gpt-4o-mini).
*   `--openai-base-url`: OpenAI API base URL.
*   `--openai-api-key`: OpenAI API key.

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

## Configuration File

Use a TOML configuration file to manage your settings.

```toml
# Example Configuration (See full documentation for options)
[babeldoc]
debug = true
lang-in = "en-US"
lang-out = "zh-CN"
openai = true
openai-model = "gpt-4o-mini"
openai-api-key = "your-api-key-here"
```

## Python API

Refer to [main.py](https://github.com/funstory-ai/yadt/blob/main/babeldoc/main.py) for example usage.  Remember to call `babeldoc.format.pdf.high_level.init()` before using the API.

*   **Offline Assets:** Use  `babeldoc.assets.assets.generate_offline_assets_package()` and `babeldoc.assets.assets.restore_offline_assets_package()` for offline asset management.

## Background

BabelDOC builds on existing work in document parsing and translation, aiming to provide a standardized pipeline for PDF processing and translation.

## Roadmap

*   Add line support
*   Add table support
*   Add cross-page/cross-column paragraph support
*   More advanced typesetting features
*   Outline support
*   ...

## Versioning

This project uses a combination of [Semantic Versioning](https://semver.org/) and [Pride Versioning](https://pridever.org/). The version number format is: "0.MAJOR.MINOR".

## Known Issues

*   Parsing errors in author and reference sections.
*   Lines are not supported.
*   Does not support drop caps.
*   Large pages may be skipped.

## How to Contribute

Contribute to BabelDOC by following the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

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