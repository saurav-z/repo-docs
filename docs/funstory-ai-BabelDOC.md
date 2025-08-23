<div align="center">
<!-- <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-banner.png" width="320px"  alt="YADT"/> -->

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
  <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
</picture>

</div>

# BabelDOC: Effortlessly Translate PDF Scientific Papers

BabelDOC is a powerful library that simplifies the process of translating PDF scientific papers. [Explore the BabelDOC Repo](https://github.com/funstory-ai/BabelDOC)

## Key Features

*   **Online Service**: Utilize the beta version on [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) for free translations (1000 pages/month).
*   **Self-Deployment**: Integrate with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for custom deployments and WebUI integration.
*   **Command-Line Interface**: Easy-to-use CLI for direct translation tasks.
*   **Python API**: Integrate BabelDOC into your Python projects.
*   **Bilingual Comparison**: Provides dual PDF output, allowing side-by-side comparison of original and translated text.

## Getting Started

### Installation (uv recommended)

1.  **Install `uv`**: Refer to [uv installation](https://github.com/astral-sh/uv#installation) and set up your `PATH`.
2.  **Install BabelDOC**:  `uv tool install --python 3.12 BabelDOC`
3.  **Run**:  `babeldoc --help` to confirm the installation.

### Installation from Source (uv recommended)

1.  **Clone the repository**: `git clone https://github.com/funstory-ai/BabelDOC`
2.  **Navigate to the directory**: `cd BabelDOC`
3.  **Install dependencies and run**: `uv run babeldoc --help`

## Advanced Options and Configuration

BabelDOC offers numerous options for fine-tuning your translation process.

### Language Options
*   `--lang-in`, `-li`: Source language code (default: en)
*   `--lang-out`, `-lo`: Target language code (default: zh)

### PDF Processing Options

*   `--files`: Input PDF documents
*   `--pages`: Specific pages to translate
*   `--split-short-lines`: split short lines into different paragraphs
*   `--short-line-split-factor`: Split threshold factor
*   `--skip-clean`: Skip PDF cleaning step
*   `--dual-translate-first`: Translated pages first in dual PDF mode
*   `--disable-rich-text-translate`: Disable rich text translation
*   `--enhance-compatibility`: Enable all compatibility enhancement options
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF
*   `--watermark-output-mode`: Control watermark output mode: 'watermarked', 'no_watermark', 'both'
*   `--max-pages-per-part`: Maximum number of pages per part for split translation
*   `--translate-table-text`: Translate table text (experimental, default: False)
*   `--formular-font-pattern`: Font pattern to identify formula text
*   `--formular-char-pattern`: Character pattern to identify formula text
*   `--show-char-box`: Show character bounding boxes (debug only)
*   `--skip-scanned-detection`: Skip scanned document detection
*   `--ocr-workaround`: Use OCR workaround
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround
*   `--primary-font-family`: Override primary font family for translated text
*   `--only-include-translated-page`: Only include translated pages in the output PDF
*   `--merge-alternating-line-numbers`: Enable post-processing to merge alternating line-number layouts
*   `--skip-form-render`: Skip form rendering
*   `--skip-curve-render`: Skip curve rendering
*   `--only-parse-generate-pdf`: Only parse PDF and generate output PDF without translation
*   `--remove-non-formula-lines`: Remove non-formula lines from paragraph areas
*   `--non-formula-line-iou-threshold`: IoU threshold for detecting paragraph overlap when removing non-formula lines
*   `--figure-table-protection-threshold`: IoU threshold for protecting lines in figure/table areas

### Translation Service Options

*   `--qps`: QPS (Queries Per Second) limit for translation service
*   `--ignore-cache`: Ignore translation cache
*   `--no-dual`: Do not output bilingual PDF files
*   `--no-mono`: Do not output monolingual PDF files
*   `--min-text-length`: Minimum text length to translate
*   `--openai`: Use OpenAI for translation
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation.
*   `--pool-max-workers`: Maximum number of worker threads
*   `--no-auto-extract-glossary`: Disable automatic term extraction

### OpenAI Specific Options

*   `--openai-model`: OpenAI model to use
*   `--openai-base-url`: Base URL for OpenAI API
*   `--openai-api-key`: API key for OpenAI service

### Glossary Options
*   `--glossary-files`: Comma-separated paths to glossary CSV files.

### Output Control

*   `--output`, `-o`: Output directory
*   `--debug`: Enable debug logging
*   `--report-interval`: Progress report interval

### General Options

*   `--warmup`: Only download and verify required assets then exit

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package
*   `--restore-offline-assets`: Restore an offline assets package

### Configuration File

*   `--config`, `-c`: Configuration file path (TOML format).

## Python API

The current recommended way to call BabelDOC in Python is to call the `high_level.do_translate_async_stream` function of [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

## Background and Architecture

BabelDOC employs a modular architecture with parsing and rendering stages. It offers an intermediate representation for flexibility in handling diverse document structures. The goal is to streamline the translation process, supporting various models, OCR engines, and rendering methods through a plugin-based system.

## Roadmap

*   Add line support
*   Add table support
*   Add cross-page/cross-column paragraph support
*   More advanced typesetting features
*   Outline support
*   ...

## Versioning

BabelDOC uses a combination of Semantic Versioning and Pride Versioning. Version format: "0.MAJOR.MINOR".

## Known Issues

1.  Parsing errors in the author and reference sections.
2.  Lines are not supported.
3.  Does not support drop caps.
4.  Large pages may be skipped.

## Contribute

Contribute to BabelDOC by following the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide. Adhere to the [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md). Contributors can receive monthly Pro membership redemption codes for Immersive Translation, see [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md).

## Acknowledgements

[See original README for list]

<h2 id="star_hist">Star History</h2>

[Star History chart]

**Important Notes:**

*   Review the [WARNING] information about `--auto-enable-ocr-workaround` in the original README.
*   Be sure to install the correct dependencies to avoid errors.