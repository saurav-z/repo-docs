# BabelDOC: Translate PDF Documents with Ease

**Instantly translate PDF scientific papers and other documents, powered by AI and open-source technology.**  [View on GitHub](https://github.com/funstory-ai/BabelDOC)

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
    <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
  </picture>

  <p>
    <a href="https://pypi.org/project/BabelDOC/">
      <img src="https://img.shields.io/pypi/v/BabelDOC">
    </a>
    <a href="https://pepy.tech/projects/BabelDOC">
      <img src="https://static.pepy.tech/badge/BabelDOC">
    </a>
    <a href="./LICENSE">
      <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC">
    </a>
    <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
      <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white">
    </a>
  </p>

  <a href="https://trendshift.io/repositories/13358" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

## Key Features

*   **PDF Translation:** Translate scientific papers and other PDF documents.
*   **Online Service:**  Try the beta version on [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) (1000 free pages/month).
*   **Self-Deployment:** Leverage [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting with a WebUI.
*   **Command Line Interface (CLI):**  Use the CLI for straightforward translation tasks.
*   **Python API:** Integrate BabelDOC into your Python projects.
*   **Bilingual PDF Output:** Create side-by-side original and translated PDFs.
*   **OpenAI Integration**: Utilize the power of OpenAI for translation, including support for compatible models like `glm-4-flash` and `deepseek-chat`.
*   **Offline Asset Management:**  Easily generate and restore offline assets for environments without internet access.

## Preview

<div align="center">
  <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## Getting Started

### Installation (Recommended: Using uv)

1.  Install [uv](https://github.com/astral-sh/uv#installation) and set up the `PATH` environment variable.
2.  Install BabelDOC:
    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```
3.  Use the `babeldoc` command:
    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here" --files example.pdf
    ```
    (For multiple files, use `--files example1.pdf --files example2.pdf`)

### Installation from Source (Recommended: Using uv)

1.  Install [uv](https://github.com/astral-sh/uv#installation) and set up the `PATH` environment variable.
2.  Clone the repository:
    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    ```
3.  Install dependencies and run BabelDOC:
    ```bash
    uv run babeldoc --help
    ```
4.  Use the `uv run babeldoc` command:
    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```
    (For multiple files, use `--files example.pdf --files example2.pdf`)

## Advanced Options

> [!NOTE]
> This CLI is primarily for debugging. End users should utilize the [Online Service](https://app.immersivetranslate.com/babel-doc/) or [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-deployment.

### Language Options

*   `--lang-in`, `-li`: Source language (default: `en`)
*   `--lang-out`, `-lo`: Target language (default: `zh`)
    >  English-to-Chinese translation is the primary focus; basic English target language support has been added.

### PDF Processing Options

*   `--files`: Input PDF files
*   `--pages`, `-p`: Specify pages to translate (e.g., "1,2,1-,-3,3-5"). If not set, translate all pages
*   `--split-short-lines`: Force split short lines into different paragraphs
*   `--short-line-split-factor`: Split threshold factor (default: 0.8)
*   `--skip-clean`: Skip PDF cleaning
*   `--dual-translate-first`: Put translated pages first in dual PDF mode (default: original pages first)
*   `--disable-rich-text-translate`: Disable rich text translation
*   `--enhance-compatibility`: Enable all compatibility enhancement options (equivalent to --skip-clean --dual-translate-first --disable-rich-text-translate)
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF. When enabled, original and translated pages are arranged in alternate order. When disabled (default), original and translated pages are shown side by side on the same page.
*   `--watermark-output-mode`: Control watermark output mode: 'watermarked' (default) adds watermark to translated PDF, 'no_watermark' doesn't add watermark, 'both' outputs both versions.
*   `--max-pages-per-part`: Maximum number of pages per part for split translation. If not set, no splitting will be performed.
*   `--no-watermark`: [DEPRECATED] Use --watermark-output-mode=no_watermark instead.
*   `--translate-table-text`: Translate table text (experimental, default: False)
*   `--formular-font-pattern`: Font pattern to identify formula text (default: None)
*   `--formular-char-pattern`: Character pattern to identify formula text (default: None)
*   `--show-char-box`: Show character bounding boxes (debug only, default: False)
*   `--skip-scanned-detection`: Skip scanned document detection (default: False). When using split translation, only the first part performs detection if not skipped.
*   `--ocr-workaround`: Use OCR workaround (default: False). Only suitable for documents with black text on white background. When enabled, white rectangular blocks will be added below the translation to cover the original text content, and all text will be forced to black color.
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False). If a document is detected as heavily scanned, this will attempt to enable OCR processing and skip further scan detection. See "Important Interaction Note" below for crucial details on how this interacts with `--ocr-workaround` and `--skip-scanned-detection`.
*   `--primary-font-family`: Override primary font family for translated text. Choices: 'serif' for serif fonts, 'sans-serif' for sans-serif fonts, 'script' for script/italic fonts. If not specified, uses automatic font selection based on original text properties.
*   `--only-include-translated-page`: Only include translated pages in the output PDF. This option is only effective when `--pages` is used. (default: False)

*   `--rpc-doclayout`: RPC service host address for document layout analysis (default: None)
*   `--working-dir`: Working directory for translation. If not set, use temp directory.
*   `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file. If not set, the glossary will not be saved.

### Translation Service Options

*   `--qps`: Queries Per Second limit (default: 4)
*   `--ignore-cache`: Ignore translation cache and force retranslation
*   `--no-dual`: Do not output bilingual PDF files
*   `--no-mono`: Do not output monolingual PDF files
*   `--min-text-length`: Minimum text length to translate (default: 5)
*   `--openai`: Use OpenAI for translation (default: False)
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation. (Currently not recommended, it may affect translation quality, default: False)
*   `--pool-max-workers`: Maximum number of worker threads for internal task processing pools. If not specified, defaults to QPS value. This parameter directly sets the worker count, replacing previous QPS-based dynamic calculations.
*   `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model to use (default: gpt-4o-mini)
*   `--openai-base-url`: Base URL for OpenAI API
*   `--openai-api-key`: API key for OpenAI service

### Glossary Options

*   `--glossary-files`: Comma-separated paths to glossary CSV files. See detailed format in the original README.

### Output Control

*   `--output`, `-o`: Output directory
*   `--debug`: Enable debug logging
*   `--report-interval`: Progress report interval in seconds (default: 0.1)

### General Options

*   `--warmup`: Download and verify assets then exit (default: False)

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package.
*   `--restore-offline-assets`: Restore an offline assets package.

## Python API

> [!TIP]
> Use the BabelDOC Python API temporarily before pdf2zh 2.0 is released; afterwards, use the pdf2zh API.  We don't provide technical support for the BabelDOC API, and API compatibility is not guaranteed.

Refer to [main.py](https://github.com/funstory-ai/yadt/blob/main/babeldoc/main.py) for an example.  Remember to call `babeldoc.format.pdf.high_level.init()` before using the API.

### Offline Asset Management in API
```python
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

## Background and Goals

This project aims to provide a standard pipeline and interface for PDF translation, leveraging parsing and rendering stages.

## Roadmap

*   [ ] Add line support
*   [ ] Add table support
*   [ ] Add cross-page/cross-column paragraph support
*   [ ] More advanced typesetting features
*   [ ] Outline support
*   [ ] ...

## Versioning

Uses Semantic Versioning with a "Pride" component:  "0.MAJOR.MINOR".

## Known Issues

*   Parsing errors in author and reference sections.
*   Lines are not supported.
*   Does not support drop caps.
*   Large pages may be skipped.

## How to Contribute

See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.  Follow the [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).

## Acknowledgements

*   [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
*   [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
*   [pdfminer](https://github.com/pdfminer/pdfminer.six)
*   [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
*   [Asynchronize](https://github.com/multimeric/Asynchronize/tree/master?tab=readme-ov-file)
*   [PriorityThreadPoolExecutor](https://github.com/oleglpts/PriorityThreadPoolExecutor)

## Star History

<a href="https://star-history.com/#funstory-ai/babeldoc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date"/>
 </picture>
</a>

> [!WARNING]
> **Important Interaction Note for `--auto-enable-ocr-workaround`:**

>  * When `--auto-enable-ocr-workaround` is set to `true`:

>  1.  During the initial setup, the values for `ocr_workaround` and `skip_scanned_detection` will be forced to `false` by `TranslationConfig`.
>  2.  If the document is heavily scanned AND `auto_enable_ocr_workaround` is `true`, the system will attempt to set both `ocr_workaround` to `true` and `skip_scanned_detection` to `true`.