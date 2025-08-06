<!-- # BabelDOC: PDF Translation Library -->

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

## BabelDOC: Effortlessly Translate Scientific PDFs with Advanced Features

BabelDOC is a powerful Python library for translating scientific PDF documents, offering both online and self-hosted solutions.  Improve your research workflow by effortlessly translating scientific papers!  Access the original repository at [funstory-ai/BabelDOC](https://github.com/funstory-ai/BabelDOC).

**Key Features:**

*   **PDF Translation:** Accurately translates scientific PDFs, preserving layout and formatting.
*   **Command-Line Interface (CLI):** Easily translate PDFs directly from your terminal.
*   **Python API:** Integrate BabelDOC into your own Python applications.
*   **Dual-Language Output:** Generate bilingual PDFs for side-by-side comparison.
*   **OpenAI Integration:** Leverage the power of OpenAI for advanced translation.
*   **Glossary Support:** Enhance accuracy with custom glossaries.
*   **Self-Deployment Options:** Deploy with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for WebUI and more translation services.
*   **Online Service:** Beta version launched [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) 1000 free pages per month.

**Preview:**

<div align="center">
<img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

**Getting Started:**

### Installation

BabelDOC can be installed using `uv` or from source.

*   **Using `uv` (Recommended):**
    1.  Install [uv](https://github.com/astral-sh/uv#installation) and set up your `PATH`.
    2.  Install BabelDOC: `uv tool install --python 3.12 BabelDOC`
    3.  Use the `babeldoc` command:  See details for the command line in the original README.

*   **From Source:**
    1.  Clone the repository: `git clone https://github.com/funstory-ai/BabelDOC`
    2.  Navigate to the project directory: `cd BabelDOC`
    3.  Install dependencies and run BabelDOC: `uv run babeldoc --help`
    4.  Use the `uv run babeldoc` command. See details for the command line in the original README.

    >  The original README provides complete instructions on using the command line, including options for OpenAI integration, file handling, and language selection.

### Use in Zotero
*   Immersive Translate Pro members can use the [immersive-translate/zotero-immersivetranslate](https://github.com/immersive-translate/zotero-immersivetranslate) plugin.
*   PDFMathTranslate self-deployed users can use the [guaguastandup/zotero-pdf2zh](https://github.com/guaguastandup/zotero-pdf2zh) plugin

### [Supported Language](https://funstory-ai.github.io/BabelDOC/supported_languages/)

## Advanced Options

> [!NOTE]
> This CLI is mainly for debugging purposes. Although end users can use this CLI to translate files, we do not provide any technical support for this purpose.
>
> End users should directly use **Online Service**: Beta version launched [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) 1000 free pages per month.
>
> End users who need self-deployment should use [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next)
> 
> If you find that an option is not listed below, it means that this option is a debugging option for maintainers. Please do not use these options.

Detailed documentation on advanced options, including language, PDF processing, translation service, OpenAI-specific settings, glossary management, output control, and offline assets management, is available in the original README.

## Python API

You can refer to the example in [main.py](https://github.com/funstory-ai/yadt/blob/main/babeldoc/main.py) to use BabelDOC's Python API.

Please note:

1. Make sure call `babeldoc.format.pdf.high_level.init()` before using the API

2. The current `TranslationConfig` does not fully validate input parameters, so you need to ensure the validity of input parameters

3. For offline assets management, you can use the following functions:
   ```python
   # Generate an offline assets package
   from pathlib import Path
   import babeldoc.assets.assets
   
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

> [!TIP]
> 
> 1. The offline assets package name cannot be modified because the file list hash is encoded in the name.
> 2. When using in production environments, it's recommended to pre-generate the assets package and include it with your application distribution.
> 3. The package verification ensures that all required assets are intact and match their expected checksums.

## Background and Roadmap

The project is built upon prior art in PDF parsing and translation, with a focus on providing a standardized pipeline. The roadmap includes expanding support for various features.

## Versioning

This project uses [Semantic Versioning](https://semver.org/) and [Pride Versioning](https://pridever.org/).

## Known Issues

See the original README for a list of known issues.

## How to Contribute

Contribute to BabelDOC!  See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide. Join the community and follow the [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).  Active contributors are eligible for [Immersive Translation](https://immersivetranslate.com) Pro membership redemption codes: [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md)

## Acknowledgements

This project is built upon the work of several open-source projects, as listed in the original README.

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