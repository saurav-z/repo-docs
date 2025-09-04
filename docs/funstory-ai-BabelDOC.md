---
# BabelDOC: Effortlessly Translate PDF Documents (and More!)

BabelDOC is your go-to library for translating PDF scientific papers, offering both online and self-hosted options to make document translation easy. [Explore the BabelDOC repository](https://github.com/funstory-ai/BabelDOC).

---

## Key Features

*   ✅ **Powerful Translation:** Translate PDF documents, with a focus on scientific papers.
*   🌐 **Online Service:** Experience BabelDOC instantly via [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/), with 1000 free pages per month.
*   🚀 **Self-Deployment:** Deploy your own instance using [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for greater control.
*   💻 **Command-Line Interface (CLI):** A user-friendly CLI for quick translation tasks.
*   🐍 **Python API:** Integrate BabelDOC functionality directly into your Python projects.
*   🌐 **Supported Languages:** Primarily designed for English to Chinese translation with support for other languages in development. ([Supported Language](https://funstory-ai.github.io/BabelDOC/supported_languages/))

## Getting Started

### Installation

#### 1.  Install using `uv`:
    *   First, install [uv](https://github.com/astral-sh/uv#installation) and set up the `PATH` environment variable as prompted.
    *   To install BabelDOC:
        ```bash
        uv tool install --python 3.12 BabelDOC
        babeldoc --help
        ```
#### 2. Install from Source
    *   First, install [uv](https://github.com/astral-sh/uv#installation) and set up the `PATH` environment variable as prompted.
    *   To install BabelDOC:
        ```bash
        git clone https://github.com/funstory-ai/BabelDOC
        cd BabelDOC
        uv run babeldoc --help
        ```

### Basic Usage

Translate a PDF file using the command line:

```bash
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf

# multiple files
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
```

## Advanced Options

(Note:  The CLI is primarily for debugging. For end-users, use the online service or PDFMathTranslate 2.0.)

### Language Options
*   `--lang-in`, `-li`: Source language code (default: en)
*   `--lang-out`, `-lo`: Target language code (default: zh)

### PDF Processing Options
*   `--files`: One or more file paths to input PDF documents.
*   `--pages`, `-p`: Specify pages to translate (e.g., "1,2,1-,-3,3-5"). If not set, translate all pages
*   `--split-short-lines`: Force split short lines into different paragraphs (may cause poor typesetting & bugs)
*   `--short-line-split-factor`: Split threshold factor (default: 0.8). The actual threshold is the median length of all lines on the current page \* this factor
*   `--skip-clean`: Skip PDF cleaning step
*   `--dual-translate-first`: Put translated pages first in dual PDF mode (default: original pages first)
*   `--disable-rich-text-translate`: Disable rich text translation (may help improve compatibility with some PDFs)
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
*   `--merge-alternating-line-numbers`: Enable post-processing to merge alternating line-number layouts (keep the number paragraph as an independent paragraph b; merge adjacent text paragraphs a and c across it when `layout_id` and `xobj_id` match, digits are ASCII and spaces only). Default: off.
*   `--skip-form-render`: Skip form rendering (default: False). When enabled, PDF forms will not be rendered in the output.
*   `--skip-curve-render`: Skip curve rendering (default: False). When enabled, PDF curves will not be rendered in the output.
*   `--only-parse-generate-pdf`: Only parse PDF and generate output PDF without translation (default: False). This skips all translation-related processing including layout analysis, paragraph finding, style processing, and translation itself. Useful for testing PDF parsing and reconstruction functionality.
*   `--remove-non-formula-lines`: Remove non-formula lines from paragraph areas (default: False). This removes decorative lines that are not part of formulas, while protecting lines in figure/table areas. Useful for cleaning up documents with decorative elements that interfere with text flow.
*   `--non-formula-line-iou-threshold`: IoU threshold for detecting paragraph overlap when removing non-formula lines (default: 0.9). Higher values are more conservative and will remove fewer lines.
*   `--figure-table-protection-threshold`: IoU threshold for protecting lines in figure/table areas when removing non-formula lines (default: 0.9). Higher values provide more protection for structural elements in figures and tables.

*   `--rpc-doclayout`: RPC service host address for document layout analysis (default: None)
*   `--working-dir`: Working directory for translation. If not set, use temp directory.
*   `--no-auto-extract-glossary`: Disable automatic term extraction. If this flag is present, the step is skipped. Defaults to enabled.
*   `--save-auto-extracted-glossary`: Save automatically extracted glossary to the specified file. If not set, the glossary will not be saved.

### Translation Service Options
*   `--qps`: QPS (Queries Per Second) limit for translation service (default: 4)
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
*   `--glossary-files`: Comma-separated paths to glossary CSV files.

### Output Control
*   `--output`, `-o`: Output directory for translated files. If not set, use current working directory.
*   `--debug`: Enable debug logging level and export detailed intermediate results in `~/.cache/yadt/working`.
*   `--report-interval`: Progress report interval in seconds (default: 0.1).

### General Options
*   `--warmup`: Only download and verify required assets then exit (default: False)

### Offline Assets Management
*   `--generate-offline-assets`: Generate an offline assets package in the specified directory. This creates a zip file containing all required models and fonts.
*   `--restore-offline-assets`: Restore an offline assets package from the specified file. This extracts models and fonts from a previously generated package.

### Configuration File
*   `--config`, `-c`: Configuration file path. Use the TOML format.

## Python API

The recommended way to call BabelDOC in Python is to call the `high_level.do_translate_async_stream` function of [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

> [!WARNING]
> **All APIs of BabelDOC should be considered as internal APIs, and any direct use of BabelDOC is not supported.**

## Background

This project offers a pipeline to translate documents with the following features:
- Parsing: A stage of parsing means to get the structure of the pdf such as text blocks, images, tables, etc.
- Rendering: A stage of rendering means to render the structure into a new pdf or other format.

## Roadmap

*   [ ] Add line support
*   [ ] Add table support
*   [ ] Add cross-page/cross-column paragraph support
*   [ ] More advanced typesetting features
*   [ ] Outline support
*   [ ] ...

## Versioning

Uses Semantic Versioning with a "MAJOR.MINOR.PATCH" format for API compatibility with [pdf2zh_next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

## Known Issues
1. Parsing errors in the author and reference sections; they get merged into one paragraph after translation.
2. Lines are not supported.
3. Does not support drop caps.
4. Large pages will be skipped.

## Contribute

Contributions are welcome! Please see the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

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