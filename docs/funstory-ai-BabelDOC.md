# BabelDOC: Effortlessly Translate PDF Documents

**Translate PDF documents into multiple languages with ease.** BabelDOC, a powerful PDF translation library, lets you convert scientific papers and other documents, providing both online and self-hosted solutions. Learn more on the [original repo](https://github.com/funstory-ai/BabelDOC)!

<div align="center">
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

## Key Features

*   **Online Service:** Access a beta version via [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) with 1000 free pages per month.
*   **Self-Deployment:** Leverage [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting, complete with a WebUI.
*   **Command Line Interface (CLI):** Easily translate PDFs using the CLI.
*   **Python API:** Integrate translation capabilities into your Python projects.
*   **Bilingual PDF Generation:** Create side-by-side original and translated PDFs for easy comparison.
*   **Offline Assets Support:** Generate and restore offline assets packages for air-gapped environments and consistent results.
*   **OpenAI Integration:** Utilize OpenAI-compatible models for high-quality translations.
*   **Glossary Support:** Enhance translation accuracy with custom glossaries.

## Getting Started

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) for virtual environment and package management.

#### Install via PyPI:

1.  Install `uv`:  Follow the instructions on the [uv installation guide](https://github.com/astral-sh/uv#installation).
2.  Install BabelDOC:

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

3.  Use the `babeldoc` command:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    ```
#### Install from Source:

1.  Clone the repository:

    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    ```

2.  Install dependencies and run BabelDOC:

    ```bash
    uv run babeldoc --help
    ```

3.  Use the `uv run babeldoc` command:

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

## Advanced Options

> [!NOTE]
> This CLI is primarily for debugging. For end-user document translation, please use the **Online Service** ([Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/)) or the self-deployable  [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

### Language Options

*   `--lang-in`, `-li`: Source language code (default: en)
*   `--lang-out`, `-lo`: Target language code (default: zh)

### PDF Processing Options

*   `--files`: Input PDF document paths.
*   `--pages`, `-p`: Specify pages to translate (e.g., "1,2,1-,-3,3-5").
*   `--split-short-lines`: Split short lines into paragraphs.
*   `--short-line-split-factor`: Split threshold factor (default: 0.8).
*   `--skip-clean`: Skip PDF cleaning step.
*   `--dual-translate-first`: Translated pages first in dual PDF mode.
*   `--disable-rich-text-translate`: Disable rich text translation.
*   `--enhance-compatibility`: Enable all compatibility enhancement options.
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF.
*   `--watermark-output-mode`: Control watermark output mode: 'watermarked' (default), 'no_watermark', 'both'.
*   `--max-pages-per-part`: Maximum pages per part for split translation.
*   `--translate-table-text`: Translate table text (experimental).
*   `--formular-font-pattern`: Font pattern to identify formula text.
*   `--formular-char-pattern`: Character pattern to identify formula text.
*   `--show-char-box`: Show character bounding boxes (debug only).
*   `--skip-scanned-detection`: Skip scanned document detection.
*   `--ocr-workaround`: Use OCR workaround.
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround.
*   `--primary-font-family`: Override primary font family.
*   `--only-include-translated-page`: Only include translated pages in output (with `--pages`).
*   `--rpc-doclayout`: RPC service host address for document layout analysis.
*   `--working-dir`: Working directory for translation.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary.

> [!TIP]
> *   Use `--enhance-compatibility` if you experience compatibility issues.
> *   Use `--max-pages-per-part` for large documents.
> *   Use `--skip-scanned-detection` if your document is not scanned.
> *   Use `--ocr-workaround` for scanned PDFs.

### Translation Service Options

*   `--qps`: QPS limit for translation service.
*   `--ignore-cache`: Ignore translation cache and force retranslation.
*   `--no-dual`: Do not output bilingual PDF files.
*   `--no-mono`: Do not output monolingual PDF files.
*   `--min-text-length`: Minimum text length to translate.
*   `--openai`: Use OpenAI for translation.
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation.
*   `--pool-max-workers`: Maximum worker threads.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.

> [!TIP]
> *   Use OpenAI-compatible LLMs.
> *   Consider models like `glm-4-flash`, `deepseek-chat`.
> *   Use `--custom-system-prompt` for instructions like Qwen 3's `/no_think`.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model to use.
*   `--openai-base-url`: Base URL for OpenAI API.
*   `--openai-api-key`: API key for OpenAI service.

### Glossary Options

*   `--glossary-files`: Comma-separated paths to glossary CSV files.

### Output Control

*   `--output`, `-o`: Output directory.
*   `--debug`: Enable debug logging.
*   `--report-interval`: Progress report interval.

### General Options

*   `--warmup`: Only download and verify assets.

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package.
*   `--restore-offline-assets`: Restore an offline assets package.

> [!TIP]
> *   Use offline assets for air-gapped environments.
> *   Generate a package once with `--generate-offline-assets`.
> *   Restore with `--restore-offline-assets`.

### Configuration File

*   `--config`, `-c`: Configuration file path (TOML format).

## Python API

```python
# Before using, make sure to call babeldoc.format.pdf.high_level.init()

# Generate an offline assets package
from pathlib import Path
import babeldoc.assets.assets

babeldoc.assets.assets.generate_offline_assets_package(Path("/path/to/output/dir"))
babeldoc.assets.assets.restore_offline_assets_package(Path("/path/to/offline_assets_package.zip"))
babeldoc.assets.assets.restore_offline_assets_package(Path("/path/to/directory"))
```

> [!TIP]
> Pre-generate the assets package for production.

## Background

BabelDOC addresses the challenges of PDF parsing and translation, providing a streamlined pipeline for converting documents.  It builds upon existing solutions like PDFMathTranslate, layoutreader, and others to offer a standardized and extensible approach.

## Roadmap

*   Add line support
*   Add table support
*   Add cross-page/cross-column paragraph support
*   More advanced typesetting features
*   Outline support
*   ...

## Version Number Explanation

This project uses a combination of [Semantic Versioning](https://semver.org/) and [Pride Versioning](https://pridever.org/).

## Known Issues

1.  Parsing errors in the author and reference sections.
2.  Lines are not supported.
3.  Does not support drop caps.
4.  Large pages may be skipped.

## How to Contribute

We welcome contributions! See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

The [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md) applies to all interactions.

## Acknowledgements

[PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), [pdfminer](https://github.com/pdfminer/pdfminer.six), [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [Asynchronize](https://github.com/multimeric/Asynchronize/tree/master?tab=readme-ov-file), [PriorityThreadPoolExecutor](https://github.com/oleglpts/PriorityThreadPoolExecutor)

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
> 1.  During the initial setup, the values for `ocr_workaround` and `skip_scanned_detection` will be forced to `false` by `TranslationConfig`, regardless of whether you also set `--ocr-workaround` or `--skip_scanned-detection` flags.
> 2.  Then, during the scanned document detection phase (`DetectScannedFile` stage):
>     *   If the document is identified as heavily scanned (e.g., >80% scanned pages) AND `auto_enable_ocr_workaround` is `true` (i.e., `translation_config.auto_enable_ocr_workaround` is true), the system will then attempt to set both `ocr_workaround` to `true` and `skip_scanned_detection` to `true`.
>
> This means that `--auto-enable-ocr-workaround` effectively gives the system control to enable OCR processing for scanned documents, potentially overriding manual settings for `--ocr-workaround` and `--skip_scanned_detection` based on its detection results. If the document is *not* detected as heavily scanned, then the initial `false` values for `ocr_workaround` and `skip_scanned_detection` (forced by `--auto-enable-ocr-workaround` at the `TranslationConfig` initialization stage) will remain in effect unless changed by other logic.