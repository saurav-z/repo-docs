<!-- # BabelDOC: PDF Translation & Bilingual Comparison Library -->

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
    <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
  </picture>

  <p>
    <a href="https://pypi.org/project/BabelDOC/">
      <img src="https://img.shields.io/pypi/v/BabelDOC" alt="PyPI version"></a>
    <a href="https://pepy.tech/projects/BabelDOC">
      <img src="https://static.pepy.tech/badge/BabelDOC" alt="Downloads"></a>
    <a href="./LICENSE">
      <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC" alt="License"></a>
    <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
      <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white" alt="Telegram"></a>
  </p>

  <a href="https://trendshift.io/repositories/13358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

## BabelDOC: Translate PDFs with Ease!

BabelDOC is a powerful Python library designed for translating PDF scientific papers and providing bilingual comparisons, offering both online and self-hosted solutions.  [Explore the BabelDOC repository](https://github.com/funstory-ai/BabelDOC).

**Key Features:**

*   **Online Translation:** Access a beta version of the online service at [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) with 1000 free pages per month.
*   **Self-Deployment:** Integrate with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting and a WebUI, expanding translation service options.
*   **Command Line Interface (CLI):** Utilize a straightforward command-line interface for quick translation tasks.
*   **Python API:** Embed translation functionality directly into your Python projects with the easy-to-use API.
*   **Dual PDF Output:**  Generate bilingual PDFs for side-by-side comparison.
*   **Offline Asset Management:** Support for offline usage and consistent results through the generation and restoration of asset packages.

> [!TIP]
>
> Integrate BabelDOC with your Zotero workflow:
>
> 1.  Use the [immersive-translate/zotero-immersivetranslate](https://github.com/immersive-translate/zotero-immersivetranslate) plugin for Immersive Translate Pro members.
> 2.  Use the [guaguastandup/zotero-pdf2zh](https://github.com/guaguastandup/zotero-pdf2zh) plugin for self-deployed PDFMathTranslate users.

[Supported Languages](https://funstory-ai.github.io/BabelDOC/supported_languages/)

## Preview

<div align="center">
  <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%" alt="BabelDOC Preview"/>
</div>

## We are hiring

See details: [EN](https://github.com/funstory-ai/jobs) | [ZH](https://github.com/funstory-ai/jobs/blob/main/README_ZH.md)

## Getting Started

### Install from PyPI

Use the [uv](https://github.com/astral-sh/uv) tool for installation.

1.  Install `uv` and set up the `PATH` environment variable as instructed in the [uv installation guide](https://github.com/astral-sh/uv#installation).
2.  Install BabelDOC using:

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

3.  Run the `babeldoc` command. Example:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf

    # multiple files
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```

### Install from Source

Use `uv` to manage virtual environments.

1.  Install `uv` and set up the `PATH` environment variable as prompted in the [uv installation guide](https://github.com/astral-sh/uv#installation).

2.  Install BabelDOC:

    ```bash
    # Clone the repository
    git clone https://github.com/funstory-ai/BabelDOC

    # Enter the project directory
    cd BabelDOC

    # Install dependencies and run babeldoc
    uv run babeldoc --help
    ```

3.  Run BabelDOC using:

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"

    # multiple files
    uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

> [!TIP]
> The use of absolute paths is recommended.

## Advanced Options

> [!NOTE]
> The CLI is primarily for debugging. For end-users, we recommend using the **Online Service**: [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) (1000 free pages/month) or, for self-deployment, [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

### Language Options

*   `--lang-in`, `-li`: Source language code (default: en)
*   `--lang-out`, `-lo`: Target language code (default: zh)

> [!TIP]
> Currently, the project focuses on English-to-Chinese translation, with expanding language support planned. (2025.3.1 update): Basic English target language support has been added. [HELP WANTED: Collecting word regular expressions for more languages](https://github.com/funstory-ai/BabelDOC/issues/129)

### PDF Processing Options

*   `--files`: Input PDF document file paths.
*   `--pages`, `-p`: Pages to translate (e.g., "1,2,1-,-3,3-5").  All pages if not set.
*   `--split-short-lines`: Split short lines into different paragraphs (may affect formatting).
*   `--short-line-split-factor`: Split threshold factor (default: 0.8).  Median line length \* this factor.
*   `--skip-clean`: Skip PDF cleaning.
*   `--dual-translate-first`: Place translated pages first in dual PDF (default: original pages first).
*   `--disable-rich-text-translate`: Disable rich text translation (improves compatibility).
*   `--enhance-compatibility`: Enable all compatibility options (`--skip-clean --dual-translate-first --disable-rich-text-translate`).
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF.
*   `--watermark-output-mode`:  Control watermark output: `'watermarked'` (default), `'no_watermark'`, `'both'`.
*   `--max-pages-per-part`:  Maximum pages per part for split translation. No splitting if not set.
*   `--no-watermark`:  [DEPRECATED] Use `--watermark-output-mode=no_watermark` instead.
*   `--translate-table-text`: Translate table text (experimental, default: False).
*   `--formular-font-pattern`: Font pattern for formula text (default: None).
*   `--formular-char-pattern`: Character pattern for formula text (default: None).
*   `--show-char-box`: Show character bounding boxes (debug only, default: False).
*   `--skip-scanned-detection`: Skip scanned document detection (default: False).
*   `--ocr-workaround`: Use OCR workaround (default: False). Black text on white background only.
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False).  See "Important Interaction Note" below.
*   `--primary-font-family`: Override primary font family (serif, sans-serif, script). Automatic if not specified.
*   `--only-include-translated-page`: Include only translated pages in output PDF (when `--pages` is used) (default: False).

*   `--rpc-doclayout`: RPC service host address for document layout analysis (default: None).
*   `--working-dir`: Working directory for translation.  Uses a temp directory if not set.
*   `--no-auto-extract-glossary`: Disable automatic term extraction (defaults to enabled).
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file.

> [!TIP]
> -  `--skip-clean` and `--dual-translate-first` can improve compatibility.
> -  `--disable-rich-text-translate` can improve compatibility.
> -  Use `--enhance-compatibility` if you have compatibility issues.
> -  Use `--max-pages-per-part` for large documents.
> -  Use `--skip-scanned-detection` for non-scanned PDFs.
> -  Use `--ocr-workaround` for scanned PDFs.

### Translation Service Options

*   `--qps`: QPS (Queries Per Second) limit (default: 4).
*   `--ignore-cache`: Force retranslation, ignoring cache.
*   `--no-dual`: Do not output bilingual PDF files.
*   `--no-mono`: Do not output monolingual PDF files.
*   `--min-text-length`: Minimum text length to translate (default: 5).
*   `--openai`: Use OpenAI for translation (default: False).
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint (not recommended, default: False).
*   `--pool-max-workers`: Maximum worker threads for internal task processing pools (defaults to QPS).
*   `--no-auto-extract-glossary`: Disable automatic term extraction (defaults to enabled).

> [!TIP]
>
> 1.  Currently, only OpenAI-compatible LLMs are supported. For more translator support, please use [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).
> 2.  Recommended models: `glm-4-flash`, `deepseek-chat`, etc.
> 3.  It is recommended to use LLMs, instead of traditional translation engines (Bing/Google).
> 4.  Use [litellm](https://github.com/BerriAI/litellm) for multiple model access.
> 5.  `--custom-system-prompt`:  Use this to add `/no_think` instruction for Qwen 3, e.g., `--custom-system-prompt "/no_think You are a professional, authentic machine translation engine."`

### OpenAI Specific Options

*   `--openai-model`: OpenAI model to use (default: gpt-4o-mini).
*   `--openai-base-url`: Base URL for OpenAI API.
*   `--openai-api-key`: API key for OpenAI service.

> [!TIP]
>
> 1.  Supports any OpenAI-compatible API (e.g., `--openai-base-url https://xxx.custom.xxx/v1`).
> 2.  For local models (Ollama), use any value for the API key (e.g., `--openai-api-key a`).

### Glossary Options

*   `--glossary-files`: Comma-separated paths to glossary CSV files.
    *   CSV format: `source`, `target`, and (optional) `tgt_lng`.
    *   `source`: Term in the original language.
    *   `target`:  Term in the target language.
    *   `tgt_lng`:  Target language code (e.g., "zh-CN", "en-US").  If included, the entry will be used only if the (normalized) `tgt_lng` matches the overall target language (`--lang-out`).  Normalization is lowercasing and replacing hyphens with underscores.  If omitted, the entry is used for any target language.
    *   Glossary names are derived from filenames.
    *   Glossaries are included in LLM prompts for term translation.

### Output Control

*   `--output`, `-o`: Output directory (current working directory if not set).
*   `--debug`: Enable debug logging.  Exports results to `~/.cache/yadt/working`.
*   `--report-interval`: Progress report interval in seconds (default: 0.1).

### General Options

*   `--warmup`: Only download and verify assets, then exit (default: False).

### Offline Assets Management

*   `--generate-offline-assets`: Generate offline assets package in the specified directory.
*   `--restore-offline-assets`: Restore an offline assets package from the specified file or directory.

> [!TIP]
>
> 1.  Offline assets are for environments without internet or to speed up installation.
> 2.  Generate with `babeldoc --generate-offline-assets /path/to/output/dir` once and distribute.
> 3.  Restore with `babeldoc --restore-offline-assets /path/to/offline_assets_*.zip`.
> 4.  Package name is fixed because the file list hash is encoded in the name.
> 5.  If you provide a directory path to `--restore-offline-assets`, the tool will automatically look for the correct offline assets package file in that directory.
> 6.  Assets packages ensure consistent results across environments.
> 7.  Generate the package on a machine with internet access first if deploying in an air-gapped environment.

### Configuration File

*   `--config`, `-c`:  Configuration file path. Uses TOML format.

Example Configuration:

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

> [!TIP]
>
> 1. Use the BabelDOC Python API temporarily before the release of pdf2zh 2.0.
> 2.  After pdf2zh 2.0 is released, use the pdf2zh API.
> 3.  The BabelDOC Python API does not guarantee compatibility.
> 4.  We do not provide technical support for the BabelDOC API.
> 5.  When developing, refer to [pdf2zh 2.0 high level](https://github.com/PDFMathTranslate/PDFMathTranslate-next/blob/main/pdf2zh_next/high_level.py) and ensure that BabelDOC runs in a subprocess.

Refer to [main.py](https://github.com/funstory-ai/yadt/blob/main/babeldoc/main.py) for an example of using the BabelDOC Python API.

Notes:

1.  Call `babeldoc.format.pdf.high_level.init()` before using the API.
2.  The `TranslationConfig` input parameters are not fully validated.
3.  Offline assets management functions:

    ```python
    # Generate an offline assets package
    from pathlib import Path
    import babeldoc.assets.assets

    # Generate package to a specific directory
    # path is optional, default is ~/.cache/babeldoc/assets/offline_assets_{hash}.zip
    babeldoc.assets.assets.generate_offline_assets_package(Path("/path/to/output/dir"))

    # Restore from a package file
    # path is optional, default is ~/.cache/babeldoc/assets/offline_assets_{hash}.zip
    babeldoc.assets.assets.restore_offline_assets_package(Path("/path/to/offline_assets_package.zip"))

    # You can also restore from a directory containing the offline assets package
    # The tool will automatically find the correct package file based on the hash
    babeldoc.assets.assets.restore_offline_assets_package(Path("/path/to/directory"))
    ```

> [!TIP]
>
> 1.  The offline assets package name cannot be modified.
> 2.  Pre-generate and include the assets package with your application for production.
> 3.  Package verification ensures assets are intact.

## Background

This project builds on existing efforts in PDF parsing and translation, including projects like [mathpix](https://mathpix.com/), [Doc2X](https://doc2x.noedgeai.com/), [minerU](https://github.com/opendatalab/MinerU), and [PDFMathTranslate](https://github.com/funstory-ai/yadt), aiming to provide a standardized pipeline and interface for PDF processing.

It addresses two core stages:

*   **Parsing:** Extracts structure from PDFs (text, images, tables).
*   **Rendering:** Generates new PDFs or formats using the extracted structure.

The project offers an intermediate representation for flexible rendering and a plugin-based system to incorporate new models, OCR engines, and renderers.

## Roadmap

*   \[ ] Add line support
*   \[ ] Add table support
*   \[ ] Add cross-page/cross-column paragraph support
*   \[ ] More advanced typesetting features
*   \[ ] Outline support
*   \[ ] ...

The initial 1.0 version aims to translate the [PDF Reference, Version 1.7](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.7old.pdf) to the following languages:

*   Simplified Chinese
*   Traditional Chinese
*   Japanese
*   Spanish

Meeting these requirements:

*   Layout error less than 1%
*   Content loss less than 1%

## Version Number Explanation

Uses Semantic Versioning and Pride Versioning.  Format: "0.MAJOR.MINOR".

*   MAJOR: Incompatible API or proud improvements.
*   MINOR:  Compatible API changes.

## Known Issues

1.  Parsing errors in author and reference sections.
2.  Line support missing.
3.  Drop caps not supported.
4.  Large pages skipped.

## How to Contribute

Contributions are welcome!  See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

Follow the YADT [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).

[Immersive Translation](https://immersivetranslate.com) sponsors monthly Pro membership redemption codes for active contributors (see: [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md)).

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
> When `--auto-enable-ocr-workaround` is set to `true`:
>
> 1.  During initial setup, `ocr_workaround` and `skip_scanned_detection` will be forced to `false` by `TranslationConfig`.
> 2.  During scanned document detection:
>     *   If the document is heavily scanned AND `auto_enable_ocr_workaround` is `true`, the system will attempt to set both `ocr_workaround` to `true` and `skip_scanned_detection` to `true`.
>
> This means that `--auto-enable-ocr-workaround` controls OCR processing for scanned documents, potentially overriding manual settings for `--ocr-workaround` and `--skip_scanned_detection` based on detection results. If the document is *not* detected as heavily scanned, the initial `false` values for `ocr_workaround` and `skip_scanned_detection` will remain unless changed by other logic.