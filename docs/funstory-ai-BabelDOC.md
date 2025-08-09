<!-- # BabelDOC: PDF Translation & Bilingual Comparison -->

<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC Dark Mode"/>
        <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
    </picture>
    <br/>
    
    <p>
        <a href="https://pypi.org/project/BabelDOC/">
            <img src="https://img.shields.io/pypi/v/BabelDOC" alt="PyPI Version"/>
        </a>
        <a href="https://pepy.tech/projects/BabelDOC">
            <img src="https://static.pepy.tech/badge/BabelDOC" alt="PyPI Downloads"/>
        </a>
        <a href="./LICENSE">
            <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC" alt="License"/>
        </a>
        <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
            <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white" alt="Telegram"/>
        </a>
    </p>

    <a href="https://trendshift.io/repositories/13358" target="_blank">
        <img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
    </a>
</div>

## BabelDOC: Instantly Translate & Compare Scientific Papers with AI

BabelDOC is a powerful library designed for translating and comparing PDF scientific papers, offering both online and self-deployment options. Dive into your research with ease, utilizing AI-powered translation to break down language barriers.  [Explore the BabelDOC repository](https://github.com/funstory-ai/BabelDOC).

**Key Features:**

*   **Online Service:**  Utilize the beta version of [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) for free, with 1000 free pages per month.
*   **Self-Deployment:**  Leverage [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting, offering a WebUI and multiple translation services.
*   **Command-Line Interface (CLI):**  A user-friendly CLI for quick translations.
*   **Python API:** Integrate BabelDOC into your Python projects for automated workflows.
*   **Dual-Language Output:** Generate bilingual PDF documents for easy comparison.
*   **OpenAI Integration:** Seamlessly utilize OpenAI for high-quality translations.
*   **Offline Asset Management:** Generate and restore offline asset packages for environments without internet access.

## Getting Started

BabelDOC can be installed via PyPI or from source.

### Install from PyPI (Recommended)

1.  Install [uv](https://github.com/astral-sh/uv#installation) and set up the `PATH` environment variable.
2.  Install BabelDOC using `uv tool install --python 3.12 BabelDOC`.
3.  Use the `babeldoc` command:

```bash
uv tool install --python 3.12 BabelDOC
babeldoc --help
```

Example usage with OpenAI:

```bash
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
```

### Install from Source

1.  Clone the repository: `git clone https://github.com/funstory-ai/BabelDOC`.
2.  Navigate to the project directory: `cd BabelDOC`.
3.  Install dependencies and run BabelDOC: `uv run babeldoc --help`.

Example usage with OpenAI:

```bash
uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
```

## Advanced Options

See the original README for detailed option explanations.

### Language Options
- `--lang-in`, `-li`: Source language code (default: en)
- `--lang-out`, `-lo`: Target language code (default: zh)

### PDF Processing Options
- `--files`: One or more file paths to input PDF documents.
- `--pages`, `-p`: Specify pages to translate (e.g., "1,2,1-,-3,3-5"). If not set, translate all pages
- `--split-short-lines`: Force split short lines into different paragraphs (may cause poor typesetting & bugs)
- `--short-line-split-factor`: Split threshold factor (default: 0.8). The actual threshold is the median length of all lines on the current page \* this factor
- `--skip-clean`: Skip PDF cleaning step
- `--dual-translate-first`: Put translated pages first in dual PDF mode (default: original pages first)
- `--disable-rich-text-translate`: Disable rich text translation (may help improve compatibility with some PDFs)
- `--enhance-compatibility`: Enable all compatibility enhancement options (equivalent to --skip-clean --dual-translate-first --disable-rich-text-translate)
- `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF. When enabled, original and translated pages are arranged in alternate order. When disabled (default), original and translated pages are shown side by side on the same page.
- `--watermark-output-mode`: Control watermark output mode: 'watermarked' (default) adds watermark to translated PDF, 'no_watermark' doesn't add watermark, 'both' outputs both versions.
- `--max-pages-per-part`: Maximum number of pages per part for split translation. If not set, no splitting will be performed.
- `--no-watermark`: [DEPRECATED] Use --watermark-output-mode=no_watermark instead.
- `--translate-table-text`: Translate table text (experimental, default: False)
- `--formular-font-pattern`: Font pattern to identify formula text (default: None)
- `--formular-char-pattern`: Character pattern to identify formula text (default: None)
- `--show-char-box`: Show character bounding boxes (debug only, default: False)
- `--skip-scanned-detection`: Skip scanned document detection (default: False). When using split translation, only the first part performs detection if not skipped.
- `--ocr-workaround`: Use OCR workaround (default: False). Only suitable for documents with black text on white background. When enabled, white rectangular blocks will be added below the translation to cover the original text content, and all text will be forced to black color.
- `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False). If a document is detected as heavily scanned, this will attempt to enable OCR processing and skip further scan detection. See "Important Interaction Note" below for crucial details on how this interacts with `--ocr-workaround` and `--skip-scanned-detection`.
- `--primary-font-family`: Override primary font family for translated text. Choices: 'serif' for serif fonts, 'sans-serif' for sans-serif fonts, 'script' for script/italic fonts. If not specified, uses automatic font selection based on original text properties.
- `--only-include-translated-page`: Only include translated pages in the output PDF. This option is only effective when `--pages` is used. (default: False)

- `--rpc-doclayout`: RPC service host address for document layout analysis (default: None)
- `--working-dir`: Working directory for translation. If not set, use temp directory.
- `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.
- `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file. If not set, the glossary will not be saved.

### Translation Service Options
- `--qps`: QPS (Queries Per Second) limit for translation service (default: 4)
- `--ignore-cache`: Ignore translation cache and force retranslation
- `--no-dual`: Do not output bilingual PDF files
- `--no-mono`: Do not output monolingual PDF files
- `--min-text-length`: Minimum text length to translate (default: 5)
- `--openai`: Use OpenAI for translation (default: False)
- `--custom-system-prompt`: Custom system prompt for translation.
- `--add-formula-placehold-hint`: Add formula placeholder hint for translation. (Currently not recommended, it may affect translation quality, default: False)
- `--pool-max-workers`: Maximum number of worker threads for internal task processing pools. If not specified, defaults to QPS value. This parameter directly sets the worker count, replacing previous QPS-based dynamic calculations.
- `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.

### OpenAI Specific Options
- `--openai-model`: OpenAI model to use (default: gpt-4o-mini)
- `--openai-base-url`: Base URL for OpenAI API
- `--openai-api-key`: API key for OpenAI service

### Glossary Options
- `--glossary-files`: Comma-separated paths to glossary CSV files.

### Output Control
- `--output`, `-o`: Output directory for translated files. If not set, use current working directory.
- `--debug`: Enable debug logging level and export detailed intermediate results in `~/.cache/yadt/working`.
- `--report-interval`: Progress report interval in seconds (default: 0.1).

### General Options
- `--warmup`: Only download and verify required assets then exit (default: False)

### Offline Assets Management
- `--generate-offline-assets`: Generate an offline assets package in the specified directory. This creates a zip file containing all required models and fonts.
- `--restore-offline-assets`: Restore an offline assets package from the specified file. This extracts models and fonts from a previously generated package.

### Configuration File
- `--config`, `-c`: Configuration file path. Use the TOML format.

## Preview

<div align="center">
    <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## Python API

Use BabelDOC's Python API to integrate translation capabilities into your projects.

**Important:**

1.  Call `babeldoc.format.pdf.high_level.init()` before using the API.
2.  Ensure the validity of input parameters, as the `TranslationConfig` doesn't fully validate them.

For offline assets management, use the following functions:

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

## We are Hiring

See details: [EN](https://github.com/funstory-ai/jobs) | [ZH](https://github.com/funstory-ai/jobs/blob/main/README_ZH.md)

## Background

BabelDOC aims to simplify the process of document translation and comparison by providing a standard pipeline and interface. It addresses the challenges of parsing and rendering PDF documents for translation, offering an intermediate representation for flexibility and plugin-based extensibility.

## Roadmap

*   Add line support
*   Add table support
*   Add cross-page/cross-column paragraph support
*   More advanced typesetting features
*   Outline support
*   ...

## Versioning

This project uses Semantic Versioning with Pride Versioning, following the format "0.MAJOR.MINOR".

## Known Issues

*   Parsing errors in author and reference sections.
*   Lines and drop caps are not supported.
*   Large pages may be skipped.

## How to Contribute

Contributions are welcome! Please refer to the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

Follow the YADT [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md) in all interactions.

[Immersive Translation](https://immersivetranslate.com) sponsors monthly Pro membership redemption codes for active contributors, see details at: [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md)

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