# BabelDOC: Effortlessly Translate PDF Scientific Papers 

Translate PDF scientific papers with ease! BabelDOC is a Python library for PDF translation and bilingual comparison. [Get started on GitHub](https://github.com/funstory-ai/BabelDOC).

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

*   **PDF Translation:** Translate PDF scientific papers, supporting English to Chinese as the primary focus.
*   **Bilingual Comparison:** Create dual-language PDFs for side-by-side comparison.
*   **Command-Line Interface (CLI):** Translate PDFs directly from your terminal.
*   **Python API:** Integrate BabelDOC into your Python projects.
*   **Online Service (Beta):** Try the [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) beta for 1000 free pages per month.
*   **Self-Deployment:** Utilize [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosted web UI with extended translation services.
*   **Offline Assets Management:** Generate and restore offline assets for use in environments without internet access.
*   **Glossary Support:** Incorporate custom glossaries for accurate translations.

## Preview

<div align="center">
    <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## We Are Hiring

See details: [EN](https://github.com/funstory-ai/jobs) | [ZH](https://github.com/funstory-ai/jobs/blob/main/README_ZH.md)

## Getting Started

### Prerequisites

*   [uv](https://github.com/astral-sh/uv#installation) (recommended)
*   Python 3.12

### Install from PyPI (Recommended)

1.  Install `uv` and set up the `PATH` environment variable.
2.  Install BabelDOC:

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

3.  Use the `babeldoc` command:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    ```

    For multiple files:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```

### Install from Source

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

    For multiple files:

    ```bash
    uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

>   **Note:** Using absolute file paths is recommended.

## Advanced Options

>   **Important:** The CLI is primarily for debugging. For end-users, use the [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) online service or [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-deployment.

### Language Options

*   `--lang-in`, `-li`: Source language code (default: `en`)
*   `--lang-out`, `-lo`: Target language code (default: `zh`)

>   **Note:** Primarily focused on English-to-Chinese translation. Basic English target language support added to minimize line breaks.

### PDF Processing Options

*   `--files`: Input PDF files.
*   `--pages`, `-p`: Pages to translate (e.g., "1,2,1-,-3,3-5").
*   `--split-short-lines`: Split short lines into paragraphs.
*   `--short-line-split-factor`: Split threshold factor (default: 0.8).
*   `--skip-clean`: Skip PDF cleaning.
*   `--dual-translate-first`: Translated pages first in dual PDF mode.
*   `--disable-rich-text-translate`: Disable rich text translation.
*   `--enhance-compatibility`: Enable all compatibility options.
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF.
*   `--watermark-output-mode`: Control watermark: `'watermarked'` (default), `'no_watermark'`, `'both'`.
*   `--max-pages-per-part`: Maximum pages per part for split translation.
*   `--translate-table-text`: Translate table text (experimental, default: `False`).
*   `--formular-font-pattern`: Font pattern for formula text.
*   `--formular-char-pattern`: Character pattern for formula text.
*   `--show-char-box`: Show character bounding boxes (debug only).
*   `--skip-scanned-detection`: Skip scanned document detection.
*   `--ocr-workaround`: Use OCR workaround for scanned PDFs.
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround.
*   `--primary-font-family`: Override primary font family.
*   `--only-include-translated-page`: Only include translated pages in output PDF (with `--pages`).
*   `--rpc-doclayout`: RPC service host address for document layout analysis.
*   `--working-dir`: Working directory for translation.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary.

>   **Tips:**
>
>   *   Use `--enhance-compatibility` for compatibility issues.
>   *   Use `--max-pages-per-part` for large documents.
>   *   Use `--skip-scanned-detection` if the document is not scanned.
>   *   Use `--ocr-workaround` for scanned PDFs.

### Translation Service Options

*   `--qps`: QPS limit for translation service (default: 4).
*   `--ignore-cache`: Ignore translation cache.
*   `--no-dual`: Do not output bilingual PDF files.
*   `--no-mono`: Do not output monolingual PDF files.
*   `--min-text-length`: Minimum text length to translate (default: 5).
*   `--openai`: Use OpenAI for translation.
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation.
*   `--pool-max-workers`: Maximum worker threads for internal task processing.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.

>   **Tips:**
>
>   1.  For more translator support, use [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).
>   2.  Use OpenAI-compatible models like `glm-4-flash`, `deepseek-chat`.
>   3.  Use [litellm](https://github.com/BerriAI/litellm) to access multiple models.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model (default: `gpt-4o-mini`).
*   `--openai-base-url`: Base URL for OpenAI API.
*   `--openai-api-key`: API key for OpenAI service.

>   **Tips:**
>
>   1.  Supports any OpenAI-compatible API endpoints.
>   2.  For local models (e.g., Ollama), use any value as the API key.

### Glossary Options

*   `--glossary-files`: Comma-separated paths to glossary CSV files.
    *   CSV columns: `source`, `target`, `tgt_lng` (optional).
    *   `tgt_lng`: Target language for specific entries (e.g., "zh-CN").

### Output Control

*   `--output`, `-o`: Output directory (current working directory if not set).
*   `--debug`: Enable debug logging.
*   `--report-interval`: Progress report interval (default: 0.1).

### General Options

*   `--warmup`: Only download and verify assets.

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package.
*   `--restore-offline-assets`: Restore an offline assets package.

>   **Tips:**
>
>   1.  Useful for offline environments.
>   2.  Generate once with `--generate-offline-assets` and distribute.
>   3.  Restore with `--restore-offline-assets`.
>   4.  Package name is encoded with a hash.
>   5.  Provide a directory to `--restore-offline-assets` for automatic package file finding.
>   6.  The package contains fonts and models to ensure consistent results.
>   7.  Package integrity is verified.
>   8.  Generate the package on a machine with internet access in air-gapped environments.

### Configuration File

*   `--config`, `-c`: Configuration file path (TOML format).

Example Configuration:

```toml
[babeldoc]
# ... (configuration options as described above) ...
```

## Python API

>   **Important:** Use BabelDOC's Python API temporarily before pdf2zh 2.0. After pdf2zh 2.0 is released, use pdf2zh's API.

*   Refer to `babeldoc/main.py` for an example.
*   Call `babeldoc.format.pdf.high_level.init()` before use.
*   Ensure input parameters are valid.
*   Use the functions for offline assets management as in the provided code sample.

## Background

This project builds upon the foundations of projects like [PDFMathTranslate](https://github.com/PDFMathTranslate/PDFMathTranslate-next) and aims to provide a standard pipeline and interface for PDF parsing and translation.

## Roadmap

*   [ ] Add line support
*   [ ] Add table support
*   [ ] Add cross-page/cross-column paragraph support
*   [ ] More advanced typesetting features
*   [ ] Outline support
*   [ ] ...

## Versioning

Uses Semantic Versioning and Pride Versioning: "0.MAJOR.MINOR".

## Known Issues

1.  Parsing errors in author/reference sections.
2.  Lines not supported.
3.  Drop caps not supported.
4.  Large pages may be skipped.

## How to Contribute

See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

Follow the YADT [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).

Contributors receive monthly Pro membership redemption codes for [Immersive Translation](https://immersivetranslate.com) - see [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md).

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

>   **Important Interaction Note for `--auto-enable-ocr-workaround`:**
>
>   When `--auto-enable-ocr-workaround` is enabled (`true`):
>
>   1.  `ocr_workaround` and `skip_scanned_detection` are forced to `false` during `TranslationConfig` initialization.
>   2.  If the document is heavily scanned and `auto_enable_ocr_workaround` is `true`, the system will set both `ocr_workaround` and `skip_scanned_detection` to `true`.