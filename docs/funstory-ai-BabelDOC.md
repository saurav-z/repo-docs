# BabelDOC: Effortlessly Translate PDF Scientific Papers

BabelDOC is your go-to library for translating PDF scientific papers, offering a user-friendly way to convert complex documents.  [Check out the original repository](https://github.com/funstory-ai/BabelDOC).

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
  <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
</picture>
</div>

## Key Features

*   **PDF Translation:** Translate scientific papers and other PDF documents.
*   **Command-Line Interface (CLI):** Easy-to-use CLI for quick translations.
*   **Python API:** Integrate BabelDOC into your Python projects seamlessly.
*   **Online Service:** Beta version available on [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) with 1000 free pages per month.
*   **Self-Deployment:** Supported by [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting with WebUI.
*   **Dual PDF Output:** Generate bilingual PDFs for side-by-side comparison.
*   **Offline Asset Management:**  Supports offline use and easier deployment in environments with limited or no internet access.
*   **OpenAI Integration:**  Uses OpenAI-compatible LLMs for advanced, high-quality translation.

## Getting Started

### Installation

We recommend using the Tool feature of [uv](https://github.com/astral-sh/uv) for package and dependency management.

1.  **Install `uv`:** Follow the instructions at [uv installation](https://github.com/astral-sh/uv#installation).
2.  **Install BabelDOC:**

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

### Usage

*   **Translate a PDF:**
    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    ```

*   **Translate Multiple PDFs:**
    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```
*   **From Source**
    ```bash
    # clone the project
    git clone https://github.com/funstory-ai/BabelDOC

    # enter the project directory
    cd BabelDOC

    # install dependencies and run babeldoc
    uv run babeldoc --help
    ```

## Advanced Options

BabelDOC offers a range of options to fine-tune your translations.  Consult the [Advanced Options](#advanced-options) section in the original README for detailed configuration.

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
- `--merge-alternating-line-numbers`: Enable post-processing to merge alternating line-number layouts (keep the number paragraph as an independent paragraph b; merge adjacent text paragraphs a and c across it when `layout_id` and `xobj_id` match, digits are ASCII and spaces only). Default: off.
- `--skip-form-render`: Skip form rendering (default: False). When enabled, PDF forms will not be rendered in the output.
- `--skip-curve-render`: Skip curve rendering (default: False). When enabled, PDF curves will not be rendered in the output.
- `--only-parse-generate-pdf`: Only parse PDF and generate output PDF without translation (default: False). This skips all translation-related processing including layout analysis, paragraph finding, style processing, and translation itself. Useful for testing PDF parsing and reconstruction functionality.
- `--remove-non-formula-lines`: Remove non-formula lines from paragraph areas (default: False). This removes decorative lines that are not part of formulas, while protecting lines in figure/table areas. Useful for cleaning up documents with decorative elements that interfere with text flow.
- `--non-formula-line-iou-threshold`: IoU threshold for detecting paragraph overlap when removing non-formula lines (default: 0.9). Higher values are more conservative and will remove fewer lines.
- `--figure-table-protection-threshold`: IoU threshold for protecting lines in figure/table areas when removing non-formula lines (default: 0.9). Higher values provide more protection for structural elements in figures and tables.

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
- `--enable-json-mode-if-requested`: Enable JSON mode for OpenAI requests (default: False)

### Glossary Options

- `--glossary-files`: Comma-separated paths to glossary CSV files.
  - Each CSV file should have the columns: `source`, `target`, and an optional `tgt_lng`.
  - The `source` column contains the term in the original language.
  - The `target` column contains the term in the target language.
  - The `tgt_lng` column (optional) specifies the target language for that specific entry (e.g., "zh-CN", "en-US").
    - If `tgt_lng` is provided for an entry, that entry will only be loaded and used if its (normalized) `tgt_lng` matches the (normalized) overall target language specified by `--lang-out`. Normalization involves lowercasing and replacing hyphens (`-`) with underscores (`_`).
    - If `tgt_lng` is omitted for an entry, that entry is considered applicable for any `--lang-out`.
  - The name of each glossary (used in LLM prompts) is derived from its filename (without the .csv extension).
  - During translation, the system will check the input text against the loaded glossaries. If terms from a glossary are found in the current text segment, that glossary (with the relevant terms) will be included in the prompt to the language model, along with an instruction to adhere to it.

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
enable-json-mode-if-requested = false  # Enable JSON mode when requested (default: false)
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

BabelDOC also offers a Python API for more advanced use cases.  See the original README or [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for more information.

## Background

BabelDOC aims to streamline document translation.  It leverages advanced parsing and rendering techniques to provide high-quality translations.

## Roadmap

The project aims to expand functionality.  Planned features include line, table, and cross-page/column support, improved typesetting, outline support, and more.

## Versioning

BabelDOC uses a combination of Semantic Versioning and Pride Versioning ("0.MAJOR.MINOR").

## Known Issues

*   Parsing errors in the author and reference sections; they get merged into one paragraph after translation.
*   Lines are not supported.
*   Does not support drop caps.
*   Large pages will be skipped.

## Contributing

Contributions are welcome! Please consult the [CONTRIBUTING guide](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md).

## Acknowledgements

Thanks to the projects and libraries listed in the original README, [listed here](https://github.com/funstory-ai/BabelDOC#acknowledgements).

## Star History

<a href="https://star-history.com/#funstory-ai/babeldoc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date"/>
 </picture>
</a>