---
title: BabelDOC: Effortlessly Translate PDF Documents with AI 🚀
description: BabelDOC is a powerful open-source library for translating scientific PDF papers, offering online and self-hosted options. Translate PDF documents with ease using AI and enjoy features like bilingual comparison and advanced typesetting.
keywords: PDF translation, AI translation, document translator, scientific papers, bilingual PDF, Python API, command line interface, open source, BabelDOC
---

# BabelDOC: AI-Powered PDF Translation 📚

BabelDOC is your go-to solution for seamless PDF scientific paper translation and bilingual comparison. Whether you need an online service or prefer self-deployment, BabelDOC empowers you to break down language barriers and understand complex documents with ease.  [Explore the BabelDOC Repository](https://github.com/funstory-ai/BabelDOC).

<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
        <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
    </picture>
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

## Key Features:

*   **AI-Powered Translation:** Leverages advanced AI models for accurate and fluent translations.
*   **Bilingual PDF Output:**  Generates side-by-side or alternating page bilingual PDFs for easy comparison.
*   **Online Service:**  Try the beta version at [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) with 1000 free pages per month.
*   **Self-Deployment:** Integrate with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for a self-hosted solution.
*   **Command Line Interface:** Provides a straightforward CLI for quick translation tasks.
*   **Python API:** Offers a flexible Python API for integration into custom workflows.
*   **Advanced Typesetting:** Preserves document structure and formatting for an improved reading experience.

## Get Started

### Installation with `uv` (Recommended)
1.  Follow the [uv installation guide](https://github.com/astral-sh/uv#installation) to install `uv`.
2.  Install BabelDOC using:
```bash
uv tool install --python 3.12 BabelDOC
babeldoc --help
```
3. Use the `babeldoc` command. For example:
```bash
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here" --files example.pdf
```
### Installation from Source
1.  Clone the repository:
```bash
git clone https://github.com/funstory-ai/BabelDOC
cd BabelDOC
```
2.  Run with `uv`:
```bash
uv run babeldoc --help
```
3.  Use the `uv run babeldoc` command. For example:
```bash
uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
```

## Advanced Options
### Configuration File
BabelDOC can be configured using a TOML configuration file.

**Example Configuration:**
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
### CLI Options

For detailed command-line options, refer to the original [README](https://github.com/funstory-ai/BabelDOC).

## Preview

<div align="center">
    <img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## Python API

Refer to the [PDFMathTranslate next](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for the currently recommended usage.

## Contribute
Help improve BabelDOC! Check out our [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.
Join our community and follow the [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).
Contributors are eligible for [Immersive Translation](https://immersivetranslate.com) Pro membership rewards: [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md)

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
```
Key improvements and explanations:

*   **SEO Optimization:** The title and description are now optimized with relevant keywords to improve search engine visibility.
*   **Concise Hook:** The first sentence acts as a hook, immediately grabbing the reader's attention and summarizing BabelDOC's core function.
*   **Clear Headings:**  Uses clear and descriptive headings to structure the information and make it easy to scan.
*   **Bulleted Key Features:**  Highlights the key features in a concise, easily digestible format.
*   **Simplified "Getting Started"**: Streamlined the installation instructions, and added direct links to online service and self-deployment to make it easier for users to find the best solution.
*   **More Informative Sections:** Added introductory text to the Python API and Advanced Options sections.
*   **Emphasis on Benefits:**  Focuses on the benefits for the user (e.g., "Effortlessly Translate," "Seamless PDF Translation").
*   **Removed Redundancy:** Removed sections that were not as important for the user.
*   **Improved Formatting**: Added more formatting to the text to make it more readable.
*   **Added Star History**: Included the star history chart to show project's popularity.