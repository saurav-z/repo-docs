# BabelDOC: Effortlessly Translate Scientific PDFs 🚀

**Translate PDFs with ease using BabelDOC, a powerful and versatile library for PDF scientific paper translation and bilingual comparison.**  ([View on GitHub](https://github.com/funstory-ai/BabelDOC))

[![PyPI Version](https://img.shields.io/pypi/v/BabelDOC)](https://pypi.org/project/BabelDOC/)
[![PyPI Downloads](https://static.pepy.tech/badge/BabelDOC)](https://pepy.tech/projects/BabelDOC)
[![License](https://img.shields.io/github/license/funstory-ai/BabelDOC)](./LICENSE)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white)](https://t.me/+Z9_SgnxmsmA5NzBl)
[![Trendshift](https://trendshift.io/api/badge/repositories/13358)](https://trendshift.io/repositories/13358)

<img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>

## Key Features

*   ✅ **Accurate Translation:**  Leverages advanced techniques for precise and context-aware translations.
*   ✅ **Bilingual PDF Output:**  Generates side-by-side or alternating page bilingual PDFs for easy comparison.
*   ✅ **Command-Line Interface (CLI):**  Translate PDFs directly from your terminal.
*   ✅ **Python API:** Integrates seamlessly into your Python projects (See [Python API](#python-api)).
*   ✅ **Online Service:** Beta version available at [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) with 1000 free pages per month.
*   ✅ **Self-Deployment:** Supports self-deployment with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).
*   ✅ **Extensive Configuration:** Offers a wide array of options for customization, including language selection, PDF processing, and translation service settings.
*   ✅ **Offline Assets Support:** Enables operation in environments without internet access, or to speed up installation using offline packages.
*   ✅ **Glossary Support:** Improve translation accuracy with glossary files.

## Getting Started

### Install with uv (Recommended)

1.  Follow the [uv installation](https://github.com/astral-sh/uv#installation) instructions to install `uv` and configure your PATH.

2.  Install BabelDOC:

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

3.  Translate your PDF:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here" --files example.pdf
    ```

### Install from Source

1.  Clone the repository:

    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    ```

2.  Install dependencies and run:

    ```bash
    uv run babeldoc --help
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

## Usage

For detailed usage instructions, including command-line options, configuration examples, and API documentation, please refer to the comprehensive sections below:

*   [Advanced Options](#advanced-options)
*   [Configuration File](#configuration-file)
*   [Python API](#python-api)

## Advanced Options

This section outlines various options available to fine-tune your PDF translation process. These include language settings, PDF processing parameters, translation service configurations (including OpenAI), glossary integration, and output control. Refer to this section for in-depth customization options.

### Language Options

*   `--lang-in`, `-li`: Source language code (default: `en`)
*   `--lang-out`, `-lo`: Target language code (default: `zh`)

### PDF Processing Options

*   `--files`: Input PDF document(s)
*   `--pages`, `-p`: Specify pages to translate (e.g., "1,2,1-,-3,3-5"). If not set, translate all pages
*   `--split-short-lines`: Force split short lines into different paragraphs
*   `--short-line-split-factor`: Split threshold factor (default: 0.8)
*   `--skip-clean`: Skip PDF cleaning step
*   `--dual-translate-first`: Put translated pages first in dual PDF mode
*   `--disable-rich-text-translate`: Disable rich text translation
*   `--enhance-compatibility`: Enable all compatibility enhancement options
*   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF
*   `--watermark-output-mode`: Control watermark output mode: 'watermarked' (default), 'no_watermark', 'both'
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
*   `--merge-alternating-line-numbers`: Merge alternating line-number layouts
*   `--skip-form-render`: Skip form rendering (default: False)
*   `--skip-curve-render`: Skip curve rendering (default: False)
*   `--only-parse-generate-pdf`: Only parse PDF and generate output PDF without translation (default: False)
*   `--remove-non-formula-lines`: Remove non-formula lines from paragraph areas (default: False)
*   `--non-formula-line-iou-threshold`: IoU threshold for detecting paragraph overlap (default: 0.9)
*   `--figure-table-protection-threshold`: IoU threshold for protecting lines in figure/table areas (default: 0.9)
*   `--rpc-doclayout`: RPC service host address for document layout analysis
*   `--working-dir`: Working directory for translation
*   `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file. If not set, the glossary will not be saved.

**Compatibility Tips:**

*   Use `--skip-clean`, `--dual-translate-first`, or `--disable-rich-text-translate` to improve compatibility with some PDF readers.
*   If you encounter compatibility issues, try `--enhance-compatibility`.

### Translation Service Options

*   `--qps`: QPS limit for translation service (default: 4)
*   `--ignore-cache`: Ignore translation cache and force retranslation
*   `--no-dual`: Do not output bilingual PDF files
*   `--no-mono`: Do not output monolingual PDF files
*   `--min-text-length`: Minimum text length to translate (default: 5)
*   `--openai`: Use OpenAI for translation (default: False)
*   `--custom-system-prompt`: Custom system prompt for translation.
*   `--add-formula-placehold-hint`: Add formula placeholder hint for translation. (Currently not recommended, it may affect translation quality, default: False)
*   `--pool-max-workers`: Maximum number of worker threads for internal task processing pools. If not specified, defaults to QPS value. This parameter directly sets the worker count, replacing previous QPS-based dynamic calculations.
*   `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.

**Recommendation**:
1.  Currently, only OpenAI-compatible LLM is supported. For more translator support, please use [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).
2.  It is recommended to use models with strong compatibility with OpenAI, such as: `glm-4-flash`, `deepseek-chat`, etc.
3.  Currently, it has not been optimized for traditional translation engines like Bing/Google, it is recommended to use LLMs.
4.  You can use [litellm](https://github.com/BerriAI/litellm) to access multiple models.
5. `--custom-system-prompt`: It is mainly used to add the `/no_think` instruction of Qwen 3 in the prompt. For example: `--custom-system-prompt "/no_think You are a professional, authentic machine translation engine."`

### OpenAI Specific Options

*   `--openai-model`: OpenAI model to use (default: gpt-4o-mini)
*   `--openai-base-url`: Base URL for OpenAI API
*   `--openai-api-key`: API key for OpenAI service

### Glossary Options

*   `--glossary-files`: Comma-separated paths to glossary CSV files.

    *   Each CSV should have `source`, `target`, and an optional `tgt_lng` column.
    *   `source`: Term in original language.
    *   `target`: Term in target language.
    *   `tgt_lng` (optional): Target language for the entry (e.g., "zh-CN").

### Output Control

*   `--output`, `-o`: Output directory for translated files
*   `--debug`: Enable debug logging
*   `--report-interval`: Progress report interval in seconds (default: 0.1)

### General Options

*   `--warmup`: Only download and verify required assets then exit

### Offline Assets Management

*   `--generate-offline-assets`: Generate an offline assets package
*   `--restore-offline-assets`: Restore an offline assets package

## Configuration File

Utilize a TOML configuration file for managing settings.

*   `--config`, `-c`: Configuration file path.

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

# Output control
no-dual = false
no-mono = false
min-text-length = 5
report-interval = 0.5
```

## Python API

The current recommended way to call BabelDOC in Python is to call the `high_level.do_translate_async_stream` function of [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

## Background & Related Projects

BabelDOC is built to provide a robust and efficient solution for scientific PDF translation. The project is inspired by and builds upon the work of several other tools and libraries designed to make document processing and translation easier.  Learn more about the underlying principles of the project and its relationship to existing solutions in the [Background](#background) section.

## Roadmap

The project's future developments will incorporate a variety of features to expand its capabilities, including:

*   Line, table, and cross-page/cross-column paragraph support.
*   Advanced typesetting features.
*   Outline support.
*   ...

## Versioning

BabelDOC uses a combination of [Semantic Versioning](https://semver.org/) and [Pride Versioning](https://pridever.org/) with the format "0.MAJOR.MINOR".

## Known Issues

Refer to the [Known Issues](#known-issues) section for details regarding the project's current limitations and potential areas for improvement.

## Contributing

We welcome contributions! See our [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide for details.

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