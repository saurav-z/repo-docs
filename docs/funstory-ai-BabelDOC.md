---
# BabelDOC: Effortlessly Translate PDF Scientific Papers

BabelDOC is your solution for seamless PDF scientific paper translation and bilingual comparison, allowing you to effortlessly translate and understand complex documents. [Visit the BabelDOC GitHub Repository](https://github.com/funstory-ai/BabelDOC) for more information and to contribute.

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
    <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
  </picture>

  <!-- Badges -->
  <p>
    <a href="https://pypi.org/project/BabelDOC/">
      <img src="https://img.shields.io/pypi/v/BabelDOC" alt="PyPI Version"></a>
    <a href="https://pepy.tech/projects/BabelDOC">
      <img src="https://static.pepy.tech/badge/BabelDOC" alt="Downloads"></a>
    <a href="./LICENSE">
      <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC" alt="License"></a>
    <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
      <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white" alt="Telegram"></a>
  </p>

  <a href="https://trendshift.io/repositories/13358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

## Key Features

*   **PDF Translation:** Translate PDF scientific papers from English to Chinese (with ongoing language support expansion).
*   **Bilingual Output:**  Generate side-by-side comparisons of original and translated text.
*   **Command-Line Interface (CLI):** Easy-to-use CLI for quick translations.
*   **Python API:** Integrate BabelDOC's functionality into your Python projects.
*   **Online Service (Beta):** Try the [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) for a free 1000-page trial.
*   **Self-Deployment:** Support via [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for advanced users.
*   **Advanced Options:** Customize translation with options for language, PDF processing, and translation service.

## Getting Started

### Installation

We recommend using the Tool feature of [uv](https://github.com/astral-sh/uv) to install BabelDOC:

1.  Install `uv` and configure the `PATH` environment variable, following the instructions in the [uv installation](https://github.com/astral-sh/uv#installation).

2.  Install BabelDOC:

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

3.  Use the `babeldoc` command, for example:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf

    # multiple files
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```

### Install from Source

We still recommend using [uv](https://github.com/astral-sh/uv) to manage virtual environments.

1.  Install `uv` and configure the `PATH` environment variable, following the instructions in the [uv installation](https://github.com/astral-sh/uv#installation).

2.  Install BabelDOC:

    ```bash
    # clone the project
    git clone https://github.com/funstory-ai/BabelDOC

    # enter the project directory
    cd BabelDOC

    # install dependencies and run babeldoc
    uv run babeldoc --help
    ```

3.  Use the `uv run babeldoc` command. For example:

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"

    # multiple files
    uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```
    > [!TIP]
    > The absolute path is recommended.

## Advanced Options

BabelDOC provides a range of options for customizing your PDF translation process. End-users are encouraged to use [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) for ease-of-use and self-deployment [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next)

### Language Options

*   `--lang-in`, `-li`: Source language code (default: en)
*   `--lang-out`, `-lo`: Target language code (default: zh)

### PDF Processing Options

*   `--files`: Input PDF document(s)
*   `--pages`, `-p`: Specify pages to translate (e.g., "1,2,1-,-3,3-5")
*   `--split-short-lines`: Force split short lines into different paragraphs
*   `--short-line-split-factor`: Split threshold factor (default: 0.8)
*   `--skip-clean`: Skip PDF cleaning step
*   `--dual-translate-first`: Put translated pages first in dual PDF mode
*   `--disable-rich-text-translate`: Disable rich text translation
*   `--enhance-compatibility`: Enable all compatibility enhancement options
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF
*   `--watermark-output-mode`: Control watermark output mode
*   `--max-pages-per-part`: Maximum number of pages per part for split translation
*   `--translate-table-text`: Translate table text (experimental, default: False)
*   `--formular-font-pattern`: Font pattern to identify formula text (default: None)
*   `--formular-char-pattern`: Character pattern to identify formula text (default: None)
*   `--show-char-box`: Show character bounding boxes (debug only, default: False)
*   `--skip-scanned-detection`: Skip scanned document detection (default: False)
*   `--ocr-workaround`: Use OCR workaround (default: False)
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False)
*   `--primary-font-family`: Override primary font family for translated text
*   `--only-include-translated-page`: Only include translated pages in the output PDF (default: False)
*   `--merge-alternating-line-numbers`: Enable post-processing to merge alternating line-number layouts (default: off)
*   `--skip-form-render`: Skip form rendering (default: False)
*   `--skip-curve-render`: Skip curve rendering (default: False)
*   `--only-parse-generate-pdf`: Only parse PDF and generate output PDF without translation (default: False)
*   `--remove-non-formula-lines`: Remove non-formula lines from paragraph areas (default: False)
*   `--non-formula-line-iou-threshold`: IoU threshold for detecting paragraph overlap when removing non-formula lines (default: 0.9)
*   `--figure-table-protection-threshold`: IoU threshold for protecting lines in figure/table areas when removing non-formula lines (default: 0.9)
*   `--rpc-doclayout`: RPC service host address for document layout analysis (default: None)
*   `--working-dir`: Working directory for translation
*   `--no-auto-extract-glossary`: Disable automatic term extraction (defaults to enabled)
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file

### Translation Service Options

*   `--qps`: QPS (Queries Per Second) limit for translation service (default: 4)
*   `--ignore-cache`: Ignore translation cache and force retranslation
*   `--no-dual`: Do not output bilingual PDF files
*   `--no-mono`: Do not output monolingual PDF files
*   `--min-text-length`: Minimum text length to translate (default: 5)
*   `--openai`: Use OpenAI for translation (default: False)
*   `--custom-system-prompt`: Custom system prompt for translation
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation (default: False)
*   `--pool-max-workers`: Maximum number of worker threads for internal task processing pools (defaults to QPS value if not set)
*   `--no-auto-extract-glossary`: Disable automatic term extraction (defaults to enabled)

### OpenAI Specific Options

*   `--openai-model`: OpenAI model to use (default: gpt-4o-mini)
*   `--openai-base-url`: Base URL for OpenAI API
*   `--openai-api-key`: API key for OpenAI service
*   `--enable-json-mode-if-requested`: Enable JSON mode for OpenAI requests (default: False)

### Glossary Options

*   `--glossary-files`: Paths to glossary CSV files

### Output Control

*   `--output`, `-o`: Output directory for translated files
*   `--debug`: Enable debug logging
*   `--report-interval`: Progress report interval in seconds (default: 0.1)

### General Options

*   `--warmup`: Only download and verify required assets then exit (default: False)

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package
*   `--restore-offline-assets`: Restore an offline assets package

## Configuration File

You can use a TOML configuration file to manage BabelDOC settings. Use the `--config` or `-c` options to specify the file path. Example configuration provided in the original README.

## Python API

The recommended way to call BabelDOC in Python is to use the `high_level.do_translate_async_stream` function of [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

## Background

BabelDOC builds upon the foundation of prior PDF processing and translation projects like [PDFMathTranslate](https://github.com/funstory-ai/yadt).

## Roadmap

See the original README for the future development plans.

## Versioning

BabelDOC uses a combination of Semantic Versioning and Pride Versioning: `0.MAJOR.MINOR`.

## Known Issues

*   See the original README for a list of known issues.

## How to Contribute

Contributions are welcome!  See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide for details.

## Acknowledgements

*   See the original README for a list of acknowledgements.

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
> See the original README for details on the interaction between `--auto-enable-ocr-workaround`, `--ocr-workaround`, and `--skip-scanned-detection`.