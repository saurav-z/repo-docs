<!-- # BabelDOC: PDF Translation & Bilingual Comparison -->

<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
        <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
    </picture>
    <br/>
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
    <a href="https://trendshift.io/repositories/13358" target="_blank">
        <img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
    </a>
</div>

## BabelDOC: Translate PDFs with Ease

**BabelDOC is a powerful library for translating PDF scientific papers and generating bilingual comparisons, offering both online and self-hosted options.**

*   **Online Service:** Experience BabelDOC with 1000 free pages per month at [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/).
*   **Self-Deployment:** Utilize [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting with a WebUI, supporting more translation services.
*   **Command-Line Interface:** A user-friendly CLI for quick translations.
*   **Python API:** Integrate BabelDOC into your Python projects for advanced customization.

>   [Original Repository](https://github.com/funstory-ai/BabelDOC)

## Key Features

*   **Accurate Translation:** Leverages advanced techniques for high-quality PDF translation.
*   **Bilingual Output:** Generates side-by-side or alternating-page bilingual PDFs for easy comparison.
*   **Language Support:** Primarily focused on English-to-Chinese translation with growing support for other languages.
*   **Customization Options:** Fine-tune translation parameters, including language selection, OCR, and glossary integration.
*   **Offline Asset Management:** Generate and restore offline asset packages for use in environments without internet access.

## Preview

<div align="center">
    <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%" alt="BabelDOC Preview"/>
</div>

## Getting Started

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) to install BabelDOC:

1.  Install `uv` following the instructions [here](https://github.com/astral-sh/uv#installation) and set up the `PATH` environment variable.
2.  Install BabelDOC:

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

3.  Use the `babeldoc` command. For example:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    # multiple files
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```

### Install from Source

1.  Clone the repository:
    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    ```
2.  Use uv to install and run BabelDOC
    ```bash
    uv run babeldoc --help
    ```
3.  Run BabelDOC.  For example:

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    # multiple files
    uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

>   **Tip:** Using absolute paths is recommended.

## Advanced Options

>   **Note:** The CLI is primarily for debugging. For end-users, use the [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) online service or [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-deployment.

### Language Options

*   `--lang-in`, `-li`: Source language code (default: en)
*   `--lang-out`, `-lo`: Target language code (default: zh)

>   **Tip:** BabelDOC primarily focuses on English-to-Chinese translation. Basic English target language support has been added.

### PDF Processing Options

*   `--files`: Input PDF file paths.
*   `--pages`, `-p`: Specify pages to translate (e.g., "1,2,1-,-3,3-5").
*   `--split-short-lines`: Force split short lines.
*   `--short-line-split-factor`: Split threshold factor (default: 0.8).
*   `--skip-clean`: Skip PDF cleaning.
*   `--dual-translate-first`: Put translated pages first in dual PDF mode (default: original pages first)
*   `--disable-rich-text-translate`: Disable rich text translation.
*   `--enhance-compatibility`: Enable all compatibility enhancement options (equivalent to --skip-clean --dual-translate-first --disable-rich-text-translate)
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF.
*   `--watermark-output-mode`: Control watermark output mode: 'watermarked' (default) adds watermark to translated PDF, 'no_watermark' doesn't add watermark, 'both' outputs both versions.
*   `--max-pages-per-part`: Maximum number of pages per part for split translation.
*   `--no-watermark`: [DEPRECATED] Use --watermark-output-mode=no_watermark instead.
*   `--translate-table-text`: Translate table text (experimental, default: False)
*   `--formular-font-pattern`: Font pattern to identify formula text (default: None)
*   `--formular-char-pattern`: Character pattern to identify formula text (default: None)
*   `--show-char-box`: Show character bounding boxes (debug only, default: False)
*   `--skip-scanned-detection`: Skip scanned document detection (default: False).
*   `--ocr-workaround`: Use OCR workaround (default: False).
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False).
*   `--primary-font-family`: Override primary font family for translated text. Choices: 'serif', 'sans-serif', 'script'.
*   `--only-include-translated-page`: Only include translated pages in the output PDF. This option is only effective when `--pages` is used. (default: False)
*   `--rpc-doclayout`: RPC service host address for document layout analysis (default: None)
*   `--working-dir`: Working directory for translation.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file.

>   **Tips:** Use `--enhance-compatibility` for compatibility issues. Use `--max-pages-per-part` for large documents.  Use `--skip-scanned-detection` if document is not a scanned PDF.  Use `--ocr-workaround` to fill background for scanned PDF.

### Translation Service Options

*   `--qps`: QPS (Queries Per Second) limit for translation service (default: 4)
*   `--ignore-cache`: Ignore translation cache.
*   `--no-dual`: Do not output bilingual PDF files.
*   `--no-mono`: Do not output monolingual PDF files.
*   `--min-text-length`: Minimum text length to translate (default: 5)
*   `--openai`: Use OpenAI for translation (default: False)
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation.
*   `--pool-max-workers`: Maximum number of worker threads.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.

>   **Tips:**  Use OpenAI-compatible LLMs. Use `--custom-system-prompt` for custom prompts.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model to use (default: gpt-4o-mini)
*   `--openai-base-url`: Base URL for OpenAI API
*   `--openai-api-key`: API key for OpenAI service

>   **Tips:** Supports any OpenAI-compatible API.

### Glossary Options

*   `--glossary-files`: Comma-separated paths to glossary CSV files.  CSV files require 'source', 'target', and optional 'tgt_lng' columns.

### Output Control

*   `--output`, `-o`: Output directory.
*   `--debug`: Enable debug logging.
*   `--report-interval`: Progress report interval.

### General Options

*   `--warmup`: Only download and verify required assets.

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package.
*   `--restore-offline-assets`: Restore an offline assets package.

>   **Tips:**  Generate a package once with `babeldoc --generate-offline-assets /path/to/output/dir` and then distribute it.  Restore the package on target machines with `babeldoc --restore-offline-assets /path/to/offline_assets_*.zip`.

### Configuration File

*   `--config`, `-c`: Configuration file path (TOML format).

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

>   **Note:** This API is for temporary use.  For a more stable API, please refer to PDFMathTranslate 2.0.  We do not provide any technical support for the BabelDOC API.

Use `babeldoc.format.pdf.high_level.init()` before using the API.
Example can be found at [main.py](https://github.com/funstory-ai/yadt/blob/main/babeldoc/main.py).

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

>   **Tips:** Pre-generate the assets package for production environments.

## Background

BabelDOC builds on existing document processing solutions such as Mathpix, Doc2X, and PDFMathTranslate.  It aims to provide a standardized pipeline and interface.

The core stages of the PDF parser/translator are parsing and rendering. BabelDOC offers an intermediate representation to handle this.

## Roadmap

*   [ ] Add line support
*   [ ] Add table support
*   [ ] Add cross-page/cross-column paragraph support
*   [ ] More advanced typesetting features
*   [ ] Outline support
*   [ ] ...

## Versioning

This project follows [Semantic Versioning](https://semver.org/) combined with [Pride Versioning](https://pridever.org/). Format is "0.MAJOR.MINOR".

## Known Issues

1.  Parsing errors in author and reference sections.
2.  Lines are not supported.
3.  Does not support drop caps.
4.  Large pages may be skipped.

## How to Contribute

Contribute to BabelDOC by following the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.  Adhere to the [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).

Contributors can earn Pro membership redemption codes from [Immersive Translation](https://immersivetranslate.com), see [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md) for details.

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

>   **Warning: Important Interaction Note for `--auto-enable-ocr-workaround`:**
>
>   When `--auto-enable-ocr-workaround` is true:
>
>   1.  `ocr_workaround` and `skip_scanned_detection` are forced to `false` during `TranslationConfig` initialization.
>   2.  If a document is heavily scanned AND `auto_enable_ocr_workaround` is true,  then the system will attempt to set both `ocr_workaround` to `true` and `skip_scanned_detection` to `true`.