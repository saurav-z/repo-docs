<!-- # BabelDOC: PDF Translation & Bilingual Comparison -->

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
    <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
  </picture>

  <p>
    <a href="https://pypi.org/project/BabelDOC/">
      <img src="https://img.shields.io/pypi/v/BabelDOC" alt="PyPI version">
    </a>
    <a href="https://pepy.tech/projects/BabelDOC">
      <img src="https://static.pepy.tech/badge/BabelDOC" alt="Downloads">
    </a>
    <a href="./LICENSE">
      <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC" alt="License">
    </a>
    <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
      <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white" alt="Telegram">
    </a>
  </p>

  <a href="https://trendshift.io/repositories/13358" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

## BabelDOC: Effortlessly Translate and Compare PDFs

BabelDOC is a powerful Python library for translating scientific papers and other PDF documents, enabling bilingual comparison and offering both online and self-hosted options.  [Explore the BabelDOC repository](https://github.com/funstory-ai/BabelDOC) to get started.

**Key Features:**

*   **Online Service:** Utilize the beta version of [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) for quick translations (1000 free pages per month).
*   **Self-Deployment:** Leverage [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting, complete with a WebUI and multiple translation service integrations.
*   **Command-Line Interface (CLI):**  A simple CLI for direct translation tasks.
*   **Python API:** Integrate BabelDOC into your own Python applications.
*   **Bilingual PDF Output:** Generates side-by-side original and translated PDFs.
*   **Customizable Options:** Control language, page selection, translation services (including OpenAI), and more.
*   **Offline Assets Management:** Generate and restore offline asset packages for use in environments without internet access.

## Preview

<div align="center">
  <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%" alt="BabelDOC Preview"/>
</div>

## We are hiring

See details: [EN](https://github.com/funstory-ai/jobs) | [ZH](https://github.com/funstory-ai/jobs/blob/main/README_ZH.md)

## Getting Started

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) for package and virtual environment management.

1.  **Install `uv`:** Follow the [uv installation instructions](https://github.com/astral-sh/uv#installation) and set up the `PATH` environment variable.

2.  **Install BabelDOC:**

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

### Usage

1.  **Translate a PDF using OpenAI (replace with your API key):**

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    ```
2. **Translate multiple PDFs**

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```
3.  **Run from Source:**

    ```bash
    # Clone the repository
    git clone https://github.com/funstory-ai/BabelDOC

    # Enter the project directory
    cd BabelDOC

    # Install dependencies and run babeldoc
    uv run babeldoc --help
    ```

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"

    # multiple files
    uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

>   **Tip:** Using absolute file paths is recommended.

## Advanced Options

### Language Options

*   `--lang-in`, `-li`: Source language code (default: en)
*   `--lang-out`, `-lo`: Target language code (default: zh)

### PDF Processing Options

*   `--files`: PDF files to translate.
*   `--pages`, `-p`: Pages to translate (e.g., "1,2,1-,-3,3-5").
*   `--split-short-lines`: Split short lines into paragraphs.
*   `--short-line-split-factor`: Split threshold factor (default: 0.8).
*   `--skip-clean`: Skip PDF cleaning step.
*   `--dual-translate-first`: Translated pages first in dual PDF mode (default: original first).
*   `--disable-rich-text-translate`: Disable rich text translation.
*   `--enhance-compatibility`: Enable all compatibility enhancement options.
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF.
*   `--watermark-output-mode`: Control watermark output mode: 'watermarked' (default), 'no_watermark', 'both'.
*   `--max-pages-per-part`: Maximum pages per part for split translation.
*   `--translate-table-text`: Translate table text (experimental, default: False)
*   `--formular-font-pattern`: Font pattern to identify formula text (default: None)
*   `--formular-char-pattern`: Character pattern to identify formula text (default: None)
*   `--show-char-box`: Show character bounding boxes (debug only, default: False)
*   `--skip-scanned-detection`: Skip scanned document detection (default: False).
*   `--ocr-workaround`: Use OCR workaround (default: False).
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False).
*   `--primary-font-family`: Override primary font family.
*   `--only-include-translated-page`: Only include translated pages in the output PDF.
*   `--rpc-doclayout`: RPC service host address for document layout analysis (default: None)
*   `--working-dir`: Working directory for translation.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary.

> **Tip:** Use `--enhance-compatibility` to resolve compatibility issues. Use `--max-pages-per-part` for large PDFs. Use `--skip-scanned-detection` when not a scanned PDF.

### Translation Service Options

*   `--qps`: QPS limit for translation service (default: 4)
*   `--ignore-cache`: Ignore translation cache.
*   `--no-dual`: Do not output bilingual PDF files
*   `--no-mono`: Do not output monolingual PDF files
*   `--min-text-length`: Minimum text length to translate (default: 5)
*   `--openai`: Use OpenAI for translation (default: False)
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation.
*   `--pool-max-workers`: Worker threads for internal task processing pools.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.

>   **Tip:** BabelDOC supports OpenAI-compatible LLMs. Use `--custom-system-prompt` for Qwen 3's `/no_think` instruction.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model (default: gpt-4o-mini)
*   `--openai-base-url`: OpenAI API base URL
*   `--openai-api-key`: OpenAI API key

>   **Tip:** BabelDOC supports any OpenAI-compatible API endpoints.

### Glossary Options

*   `--glossary-files`: Paths to glossary CSV files.  Files should include 'source', 'target', and optional 'tgt_lng' columns.

### Output Control

*   `--output`, `-o`: Output directory.
*   `--debug`: Enable debug logging.
*   `--report-interval`: Progress report interval (default: 0.1).

### General Options

*   `--warmup`: Only download and verify required assets then exit (default: False)

### Offline Assets Management

*   `--generate-offline-assets`: Generate offline assets package.
*   `--restore-offline-assets`: Restore offline assets package.

>   **Tip:** Create offline packages for environments without internet. The package file name cannot be modified.

### Configuration File

*   `--config`, `-c`: Configuration file path (TOML format). See the example configuration in the original README.

## Python API

Use the Python API for integration into your applications.  Consult the example in `babeldoc/main.py`.

>   **Important:** Before the release of pdf2zh 2.0, you can use BabelDOC's Python API.  After the release of pdf2zh 2.0, please use its Python API instead.  The BabelDOC API does not guarantee compatibility.

1.  Make sure to call `babeldoc.format.pdf.high_level.init()` before using the API.
2.  The current `TranslationConfig` does not fully validate the input parameters; ensure their validity.
3.  Offline Assets Management: (See original README for example code.)

>   **Tip:** The offline assets package name is encoded. Pre-generate and include the assets in production environments.

## Background

(Summarized from original README)

BabelDOC builds on existing PDF parsing and translation projects like PDFMathTranslate and others. It offers a standardized pipeline for PDF structure parsing and rendering, allowing for modular integration of different models and renderers.  The pipeline includes parsing, rendering, and an intermediate representation, enabling the creation of bilingual PDFs and addressing the limitations of existing solutions.

## Roadmap

(Summarized from original README)

The initial 1.0 version aims to accurately translate PDF Reference, Version 1.7, into simplified Chinese, traditional Chinese, Japanese, and Spanish. Key objectives include layout and content accuracy.

## Version Number Explanation

BabelDOC uses Semantic Versioning with Pride Versioning: "0.MAJOR.MINOR".

*   `MAJOR`: API incompatible changes or proud improvements.
*   `MINOR`: API compatible changes.

## Known Issues

(Summarized from original README)

*   Parsing errors in the author and reference sections.
*   Lines are not fully supported.
*   Does not support drop caps.
*   Large pages can be skipped.

## How to Contribute

Contributions are welcome! See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.  Follow the YADT [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).

Contributors may receive monthly Pro membership redemption codes for Immersive Translation.  See [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md).

## Acknowledgements

(Summarized from original README)

Includes a list of projects that the BabelDOC project uses.

<h2 id="star_hist">Star History</h2>

<a href="https://star-history.com/#funstory-ai/babeldoc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date"/>
 </picture>
</a>

>   **Important Interaction Note for `--auto-enable-ocr-workaround`:**

>   (See original README for details.)