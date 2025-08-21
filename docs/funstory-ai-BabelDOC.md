# BabelDOC: Translate PDF Documents with Ease

**BabelDOC is your go-to library for effortlessly translating PDF scientific papers and documents.**  [Visit the original repository on GitHub](https://github.com/funstory-ai/BabelDOC)

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
    <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
  </picture>
  <p>
    <a href="https://pypi.org/project/BabelDOC/">
      <img src="https://img.shields.io/pypi/v/BabelDOC"></a>
    <a href="https://pepy.tech/projects/BabelDOC">
      <img src="https://static.pepy.tech/badge/BabelDOC"></a>
    <a href="./LICENSE">
      <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC"></a>
    <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
      <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white"></a>
  </p>
  <a href="https://trendshift.io/repositories/13358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

## Key Features

*   **Scientific Paper Translation:**  Translate PDF documents, especially scientific papers.
*   **Bilingual Comparison:** Generate side-by-side bilingual PDF files for easy comparison.
*   **Online Service:** Utilize the [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) beta for 1000 free pages per month.
*   **Self-Deployment:** Integrate BabelDOC with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting.
*   **Command-Line Interface (CLI):**  A straightforward CLI for basic translation tasks.
*   **Python API:**  Embed translation capabilities directly into your Python applications.
*   **Offline Assets Management:** Easily manage dependencies for environments without internet access.

## Getting Started

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) to install and manage BabelDOC.

**1. Install uv:** Follow the instructions on the [uv installation page](https://github.com/astral-sh/uv#installation) to install `uv` and set up your `PATH` environment variable.

**2. Install BabelDOC using uv:**

   ```bash
   uv tool install --python 3.12 BabelDOC
   babeldoc --help
   ```

### Installation from Source
We still recommend using [uv](https://github.com/astral-sh/uv) to manage virtual environments.

**1. Clone the Repository**
```bash
git clone https://github.com/funstory-ai/BabelDOC
cd BabelDOC
```

**2. Install Dependencies**
```bash
uv run babeldoc --help
```
**3. Use the `uv run babeldoc` command.**
```bash
uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"

# multiple files
uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
```

### Examples
*   `babeldoc --files my_document.pdf --lang-in en --lang-out zh --openai --openai-api-key YOUR_OPENAI_API_KEY`
*   `babeldoc --files paper1.pdf paper2.pdf --pages 1-3,5 --no-dual`

## Advanced Options

### Language Options
-   `--lang-in`, `-li`: Source language code (default: en)
-   `--lang-out`, `-lo`: Target language code (default: zh)

### PDF Processing Options
-   `--files`: One or more file paths to input PDF documents.
-   `--pages`, `-p`: Specify pages to translate (e.g., "1,2,1-,-3,3-5"). If not set, translate all pages
-   `--split-short-lines`: Force split short lines into different paragraphs (may cause poor typesetting & bugs)
-   `--short-line-split-factor`: Split threshold factor (default: 0.8). The actual threshold is the median length of all lines on the current page \* this factor
-   `--skip-clean`: Skip PDF cleaning step
-   `--dual-translate-first`: Put translated pages first in dual PDF mode (default: original pages first)
-   `--disable-rich-text-translate`: Disable rich text translation (may help improve compatibility with some PDFs)
-   `--enhance-compatibility`: Enable all compatibility enhancement options (equivalent to --skip-clean --dual-translate-first --disable-rich-text-translate)
-   `--use-alternating-pages-dual`: Use alternating pages mode for dual PDF. When enabled, original and translated pages are arranged in alternate order. When disabled (default), original and translated pages are shown side by side on the same page.
-   `--watermark-output-mode`: Control watermark output mode: 'watermarked' (default) adds watermark to translated PDF, 'no_watermark' doesn't add watermark, 'both' outputs both versions.
-   `--max-pages-per-part`: Maximum number of pages per part for split translation. If not set, no splitting will be performed.
-   `--no-watermark`: [DEPRECATED] Use --watermark-output-mode=no_watermark instead.
-   `--translate-table-text`: Translate table text (experimental, default: False)
-   `--formular-font-pattern`: Font pattern to identify formula text (default: None)
-   `--formular-char-pattern`: Character pattern to identify formula text (default: None)
-   `--show-char-box`: Show character bounding boxes (debug only, default: False)
-   `--skip-scanned-detection`: Skip scanned document detection (default: False). When using split translation, only the first part performs detection if not skipped.
-   `--ocr-workaround`: Use OCR workaround (default: False). Only suitable for documents with black text on white background. When enabled, white rectangular blocks will be added below the translation to cover the original text content, and all text will be forced to black color.
-   `--auto-enable-ocr-workaround`: Enable automatic OCR workaround (default: False). If a document is detected as heavily scanned, this will attempt to enable OCR processing and skip further scan detection. See "Important Interaction Note" below for crucial details on how this interacts with `--ocr-workaround` and `--skip-scanned-detection`.
-   `--primary-font-family`: Override primary font family for translated text. Choices: 'serif' for serif fonts, 'sans-serif' for sans-serif fonts, 'script' for script/italic fonts. If not specified, uses automatic font selection based on original text properties.
-   `--only-include-translated-page`: Only include translated pages in the output PDF. This option is only effective when `--pages` is used. (default: False)

-   `--rpc-doclayout`: RPC service host address for document layout analysis (default: None)
-   `--working-dir`: Working directory for translation. If not set, use temp directory.
-   `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.
-   `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file. If not set, the glossary will not be saved.

### Translation Service Options
-   `--qps`: QPS (Queries Per Second) limit for translation service (default: 4)
-   `--ignore-cache`: Ignore translation cache and force retranslation
-   `--no-dual`: Do not output bilingual PDF files
-   `--no-mono`: Do not output monolingual PDF files
-   `--min-text-length`: Minimum text length to translate (default: 5)
-   `--openai`: Use OpenAI for translation (default: False)
-   `--custom-system-prompt`: Custom system prompt for translation.
-   `--add-formula-placehold-hint`: Add formula placeholder hint for translation. (Currently not recommended, it may affect translation quality, default: False)
-   `--pool-max-workers`: Maximum number of worker threads for internal task processing pools. If not specified, defaults to QPS value. This parameter directly sets the worker count, replacing previous QPS-based dynamic calculations.
-   `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.

### OpenAI Specific Options
-   `--openai-model`: OpenAI model to use (default: gpt-4o-mini)
-   `--openai-base-url`: Base URL for OpenAI API
-   `--openai-api-key`: API key for OpenAI service

### Glossary Options
-   `--glossary-files`: Comma-separated paths to glossary CSV files.
  -   Each CSV file should have the columns: `source`, `target`, and an optional `tgt_lng`.
  -   The `source` column contains the term in the original language.
  -   The `target` column contains the term in the target language.
  -   The `tgt_lng` column (optional) specifies the target language for that specific entry (e.g., "zh-CN", "en-US").
    -   If `tgt_lng` is provided for an entry, that entry will only be loaded and used if its (normalized) `tgt_lng` matches the (normalized) overall target language specified by `--lang-out`. Normalization involves lowercasing and replacing hyphens (`-`) with underscores (`_`).
    -   If `tgt_lng` is omitted for an entry, that entry is considered applicable for any `--lang-out`.
  -   The name of each glossary (used in LLM prompts) is derived from its filename (without the .csv extension).
  -   During translation, the system will check the input text against the loaded glossaries. If terms from a glossary are found in the current text segment, that glossary (with the relevant terms) will be included in the prompt to the language model, along with an instruction to adhere to it.

### Output Control
-   `--output`, `-o`: Output directory for translated files. If not set, use current working directory.
-   `--debug`: Enable debug logging level and export detailed intermediate results in `~/.cache/yadt/working`.
-   `--report-interval`: Progress report interval in seconds (default: 0.1).

### General Options
-   `--warmup`: Only download and verify required assets then exit (default: False)

### Offline Assets Management
-   `--generate-offline-assets`: Generate an offline assets package in the specified directory. This creates a zip file containing all required models and fonts.
-   `--restore-offline-assets`: Restore an offline assets package from the specified file. This extracts models and fonts from a previously generated package.

### Configuration File
-   `--config`, `-c`: Configuration file path. Use the TOML format.

### Example Configuration

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

## Preview

<div align="center">
  <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## Python API

```python
from babeldoc.format.pdf.high_level import init
from babeldoc.config import TranslationConfig
from pathlib import Path
from babeldoc.translate import translate_pdf

# Initialize BabelDOC
init()

# Configuration
config = TranslationConfig(
    files=[Path("example.pdf")],
    lang_in="en",
    lang_out="zh",
    openai=True,
    openai_api_key="YOUR_OPENAI_API_KEY",
    output=Path("./output")
)

# Translate
translate_pdf(config)
```

### Offline Assets Management
```python
from pathlib import Path
import babeldoc.assets.assets

# Generate an offline assets package
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

## Contributions

We welcome contributions!  See our [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide for details.  Please adhere to the [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).

Contributors are eligible for [Immersive Translation](https://immersivetranslate.com) Pro membership redemption codes, learn more in [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md).

## Acknowledgements

*   [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
*   [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
*   [pdfminer](https://github.com/pdfminer/pdfminer.six)
*   [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
*   [Asynchronize](https://github.com/multimeric/Asynchronize/tree/master?tab=readme-ov-file)
*   [PriorityThreadPoolExecutor](https://github.com/oleglpts/PriorityThreadPoolExecutor)

## Star History

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