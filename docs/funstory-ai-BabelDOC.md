<!-- # BabelDOC: Effortlessly Translate PDF Documents -->

<div align="center">
  <!-- Logo with dark mode support -->
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC (Dark Mode)"/>
    <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
  </picture>

  <!-- PyPI Badges and other badges -->
  <p>
    <a href="https://pypi.org/project/BabelDOC/">
      <img src="https://img.shields.io/pypi/v/BabelDOC" alt="PyPI Version">
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

## BabelDOC: Translate PDFs with Ease

BabelDOC is a powerful Python library designed for translating scientific papers and other PDF documents, offering online and self-hosted options. 
[Explore the BabelDOC repository](https://github.com/funstory-ai/BabelDOC) to revolutionize your document translation workflow.

**Key Features:**

*   **Online Service:** Access a beta version through [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) with 1000 free pages per month.
*   **Self-Deployment:** Integrate with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting and a web UI with various translation services.
*   **Command-Line Interface (CLI):** Utilize a simple and effective CLI for direct translation tasks.
*   **Python API:** Integrate BabelDOC functionality into your Python projects.
*   **Dual PDF Output:** Generate bilingual PDFs for side-by-side comparison.
*   **Extensive Customization:**  Customize translation behavior with numerous options for language, PDF processing, translation services, and output control.

> **Tip:**  For Zotero users, leverage the [immersive-translate/zotero-immersivetranslate](https://github.com/immersive-translate/zotero-immersivetranslate) plugin (Immersive Translate Pro members) or the [guaguastandup/zotero-pdf2zh](https://github.com/guaguastandup/zotero-pdf2zh) plugin (PDFMathTranslate self-deployed users).

## Preview

<div align="center">
  <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%" alt="BabelDOC Preview"/>
</div>

## Hiring

We are hiring! See details: [EN](https://github.com/funstory-ai/jobs) | [ZH](https://github.com/funstory-ai/jobs/blob/main/README_ZH.md)

## Getting Started

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) for package and virtual environment management.

#### Install via PyPI

1.  Follow the [uv installation instructions](https://github.com/astral-sh/uv#installation) to install uv and set up your `PATH` environment variable.
2.  Install BabelDOC using:

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```
3. Use the `babeldoc` command. For example:

```bash
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf

# multiple files
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
```

#### Install from Source

1.  Follow the [uv installation instructions](https://github.com/astral-sh/uv#installation) to install uv and set up your `PATH` environment variable.
2.  Clone the repository and navigate to the project directory:

    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    ```
3. Install dependencies with:

    ```bash
    uv run babeldoc --help
    ```
4. Use the `uv run babeldoc` command. For example:

```bash
uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"

# multiple files
uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
```

> **Recommendation:** Always specify the absolute path to your PDF files.

## Advanced Options

> [!NOTE]
> The CLI is primarily for debugging. We recommend end-users use the **Online Service** or **PDFMathTranslate 2.0** for self-deployment.

### Language Options

*   `--lang-in`, `-li`: Source language code (default: en)
*   `--lang-out`, `-lo`: Target language code (default: zh)

> **Note:** Primarily designed for English-to-Chinese translation. English target language support has been added to minimize line breaks.

### PDF Processing Options

*   `--files`: Input PDF file paths.
*   `--pages`, `-p`: Pages to translate (e.g., "1,2,1-,-3,3-5"). Defaults to all pages.
*   `--split-short-lines`: Split short lines into paragraphs (may impact formatting).
*   `--short-line-split-factor`: Split threshold factor (default: 0.8).
*   `--skip-clean`: Skip PDF cleaning.
*   `--dual-translate-first`:  Translated pages first in dual PDF mode.
*   `--disable-rich-text-translate`: Disable rich text translation.
*   `--enhance-compatibility`: Enable all compatibility enhancements.
*   `--use-alternating-pages-dual`: Alternate original and translated pages in dual PDF.
*   `--watermark-output-mode`: Control watermark: 'watermarked' (default), 'no_watermark', 'both'.
*   `--max-pages-per-part`: Maximum pages per part for split translation.
*   `--translate-table-text`: Translate table text (experimental, default: False).
*   `--formular-font-pattern`: Font pattern to identify formula text.
*   `--formular-char-pattern`: Character pattern to identify formula text.
*   `--show-char-box`: Show character bounding boxes (debug only).
*   `--skip-scanned-detection`: Skip scanned document detection.
*   `--ocr-workaround`: Use OCR workaround.
*   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround. See "Important Interaction Note" below.
*   `--primary-font-family`: Override primary font family for translated text.
*   `--only-include-translated-page`: Only include translated pages in output PDF. Effective when `--pages` is used.
*   `--merge-alternating-line-numbers`: Merge alternating line-number layouts.
*   `--skip-form-render`: Skip form rendering.
*   `--skip-curve-render`: Skip curve rendering.
*   `--only-parse-generate-pdf`: Only parse and generate PDF without translation.
*   `--remove-non-formula-lines`: Remove non-formula lines from paragraph areas.
*   `--non-formula-line-iou-threshold`: IoU threshold for detecting paragraph overlap when removing non-formula lines.
*   `--figure-table-protection-threshold`: IoU threshold for protecting lines in figure/table areas.

*   `--rpc-doclayout`: RPC service host address for document layout analysis.
*   `--working-dir`: Working directory for translation.
*   `--no-auto-extract-glossary`: Disable automatic term extraction.
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary.

> **Tips:**
>
> *   `--skip-clean`, `--dual-translate-first`, and `--disable-rich-text-translate` may improve compatibility.
> *   Use `--enhance-compatibility` for general compatibility issues.
> *   Use `--max-pages-per-part` for large documents.
> *   Use `--skip-scanned-detection` if your document is not scanned.
> *   Use `--ocr-workaround` for scanned PDFs.

### Translation Service Options

*   `--qps`: Queries Per Second limit (default: 4).
*   `--ignore-cache`: Force retranslation.
*   `--no-dual`: Do not output bilingual PDF files.
*   `--no-mono`: Do not output monolingual PDF files.
*   `--min-text-length`: Minimum text length to translate (default: 5).
*   `--openai`: Use OpenAI for translation (default: False).
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint (not recommended, default: False).
*   `--pool-max-workers`: Worker thread count (defaults to QPS).
*   `--no-auto-extract-glossary`: Disable automatic term extraction.

> **Tips:**
>
> 1.  Currently supports OpenAI-compatible LLMs. Use [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for more translation services.
> 2.  Recommended models: `glm-4-flash`, `deepseek-chat`, etc.
> 3.  It's recommended to use LLMs, traditional engines like Bing/Google are not optimized.
> 4.  You can use [litellm](https://github.com/BerriAI/litellm) to access multiple models.
> 5.  `--custom-system-prompt` is used for adding `/no_think` Qwen 3 instruction.

### OpenAI Specific Options

*   `--openai-model`: OpenAI model to use (default: gpt-4o-mini).
*   `--openai-base-url`: OpenAI API base URL.
*   `--openai-api-key`: OpenAI API key.

> **Tips:**
>
> 1.  Supports any OpenAI-compatible API endpoints (e.g., `https://xxx.custom.xxx/v1`).
> 2.  For local models (e.g., Ollama), use any value for the API key.

### Glossary Options

*   `--glossary-files`: Comma-separated paths to glossary CSV files.
    *   Each CSV file must have `source`, `target`, and an optional `tgt_lng` columns.

### Output Control

*   `--output`, `-o`: Output directory (defaults to current directory).
*   `--debug`: Enable debug logging.
*   `--report-interval`: Progress report interval (default: 0.1).

### General Options

*   `--warmup`: Download assets and exit.

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package.
*   `--restore-offline-assets`: Restore an offline assets package.

> **Tips:**
>
> 1.  Useful for environments without internet.
> 2.  Generate the package with `--generate-offline-assets`.
> 3.  Restore with `--restore-offline-assets`.
> 4.  Package name is encoded with a file list hash, do not modify it.
> 5.  The tool will automatically find the package in a specified directory.
> 6.  Ensures consistent results across different environments by including fonts and models.
> 7.  Verify the integrity of all assets using SHA3-256 hashes.
> 8. Generate the package on a machine with internet access before deploying to air-gapped environments.

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
skip_form_render = false # Skip form rendering (default: False)
skip_curve_render = false # Skip curve rendering (default: False)
only_parse_generate_pdf = false # Only parse PDF and generate output PDF without translation (default: False)
remove_non_formula_lines = false # Remove non-formula lines from paragraph areas (default: False)
non_formula_line_iou_threshold = 0.2 # IoU threshold for paragraph overlap detection (default: 0.2)
figure_table_protection_threshold = 0.3 # IoU threshold for figure/table protection (default: 0.3)

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

Refer to `high_level.do_translate_async_stream` in [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for Python API use.

> **Warning:**  APIs of BabelDOC are considered internal and direct use is not supported.

## Background

[Detailed background information is provided in the original README.]

## Roadmap

*   [ ] Add line support
*   [ ] Add table support
*   [ ] Add cross-page/cross-column paragraph support
*   [ ] More advanced typesetting features
*   [ ] Outline support
*   [ ] ...

## Versioning

Uses Semantic Versioning and Pride Versioning (0.MAJOR.MINOR).

> [!NOTE]
> The API compatibility refers to compatibility with [pdf2zh_next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

-   MAJOR: API incompatible changes or proud improvements.
-   MINOR: API compatible changes.

## Known Issues

[Listing of known issues from the original README.]

## How to Contribute

[Instructions from the original README, including the link to CONTRIBUTING and CODE_OF_CONDUCT]

[Immersive Translation sponsors monthly Pro membership for active contributors, see CONTRIBUTOR_REWARD.md]

## Acknowledgements

[List of acknowledgements from the original README.]

<h2 id="star_hist">Star History</h2>

[Star History Chart]

> [!WARNING]
> **Important Interaction Note for `--auto-enable-ocr-workaround`:**
>
> [See original README for details.]