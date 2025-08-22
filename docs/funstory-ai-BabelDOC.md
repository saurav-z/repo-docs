<!-- # BabelDOC: The Ultimate PDF Translation Library -->

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
    <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
  </picture>

  <p>
    <!-- PyPI -->
    <a href="https://pypi.org/project/BabelDOC/">
      <img src="https://img.shields.io/pypi/v/BabelDOC"></a>
    <a href="https://pepy.tech/projects/BabelDOC">
      <img src="https://static.pepy.tech/badge/BabelDOC"></a>
    <!-- License -->
    <a href="./LICENSE">
      <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC"></a>
    <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
      <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white"></a>
  </p>

  <a href="https://trendshift.io/repositories/13358" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

## BabelDOC: Effortlessly Translate Scientific Papers and PDFs

BabelDOC is a powerful Python library designed to translate PDF scientific papers, providing bilingual comparisons and supporting various features for seamless document translation.  [View the source on GitHub](https://github.com/funstory-ai/BabelDOC).

**Key Features:**

*   **PDF Translation:**  Translate PDF documents efficiently.
*   **Online Service:** Beta version available on [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) with 1000 free pages per month.
*   **Self-Deployment:** Integrated with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting and WebUI with additional translation services.
*   **Command Line Interface (CLI):**  Easy-to-use CLI for quick translations.
*   **Python API:** Integrate translation capabilities directly into your Python projects.
*   **Glossary Support:** Improve translation accuracy using custom glossaries.
*   **Offline Assets Management:** Generate and restore offline asset packages for environments with limited or no internet access.
*   **Advanced PDF Processing Options:** Control page selection, splitting, and compatibility settings.
*   **OpenAI Integration:** Support for OpenAI-compatible LLMs.

## Getting Started

### Installation

**Recommended:** Utilize [uv](https://github.com/astral-sh/uv) for managing your virtual environment.

1.  **Install `uv`:**  Follow the [uv installation instructions](https://github.com/astral-sh/uv#installation).

2.  **Install BabelDOC:**

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

### Using the CLI

*   **Translate a PDF:**

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    ```

*   **Translate Multiple PDFs:**

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```

### Installation from Source

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    ```

2.  **Install Dependencies and Run:**

    ```bash
    uv run babeldoc --help
    ```

*   **Using from Source**

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```
    ```bash
    uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

## Advanced Options

For detailed information on advanced options, refer to the original [README](https://github.com/funstory-ai/BabelDOC).

## Python API

For using BabelDOC as a library within your Python projects, you can refer to the example in [main.py](https://github.com/funstory-ai/yadt/blob/main/babeldoc/main.py).

## Offline Assets Management

*   **Generate Package:**  `babeldoc --generate-offline-assets /path/to/output/dir`
*   **Restore Package:**  `babeldoc --restore-offline-assets /path/to/offline_assets_*.zip` or `babeldoc --restore-offline-assets /path/to/directory`

## Background and Roadmap

BabelDOC aims to provide a robust and flexible solution for PDF translation.

*   **Roadmap:**
    *   Add line support
    *   Add table support
    *   Add cross-page/cross-column paragraph support
    *   More advanced typesetting features
    *   Outline support

## Versioning and Contribution

*   **Versioning:** Uses a combination of Semantic Versioning and Pride Versioning (0.MAJOR.MINOR).
*   **Contributing:**  We welcome contributions! See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

## Acknowledgements

*   PDFMathTranslate
*   DocLayout-YOLO
*   pdfminer
*   PyMuPDF
*   Asynchronize
*   PriorityThreadPoolExecutor

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
```
Key improvements and explanations:

*   **SEO Optimization:**
    *   **Clear Title and Hook:**  "BabelDOC: Effortlessly Translate Scientific Papers and PDFs" immediately grabs attention and includes key search terms.
    *   **Keyword Placement:**  Keywords like "PDF translation," "scientific papers," "Python library," and "bilingual comparison" are strategically placed throughout the README.
    *   **Headings:** Using clear headings (`Getting Started`, `Key Features`, `Advanced Options`, `Python API`, etc.) makes the content scannable and improves SEO.
*   **Summarization and Clarity:**
    *   The introduction is concise and clearly defines BabelDOC's purpose.
    *   Information is reorganized and presented in a more user-friendly way, using bullet points and shorter paragraphs.
    *   Irrelevant content like the author and reference sections in the "Background" section has been removed.
*   **Structure:**
    *   **Table of Contents (Implicit):** The use of headings effectively creates an implicit table of contents, guiding the reader through the information.
*   **Actionable Instructions:**
    *   Clear installation instructions are provided, with a recommended tool (uv).
    *   Example CLI usage is given for immediate usability.
*   **Emphasis on Key Features:**
    *   The "Key Features" section immediately highlights the most important aspects of BabelDOC.
*   **Conciseness:** Redundant phrases have been removed, and information is presented efficiently.
*   **Call to Action (Contribution):** Encourages contributions and provides links to relevant documentation.
*   **Updated Sections:** The document includes the important interaction notes for `--auto-enable-ocr-workaround`.
*   **Removed non-essential sections:** The original README contains sections that are not vital for the core documentation (e.g. "Known Issues" and "Background").  These have been omitted for conciseness, as this is a simplified overview. These can be placed into seperate files.
*   **Clear Instructions to View the Source:** Includes the link to github.