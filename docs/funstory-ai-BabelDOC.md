# BabelDOC: Translate PDF Documents with Ease

**BabelDOC effortlessly translates PDF scientific papers, offering both online and self-hosted solutions for seamless multilingual document access. ([See Original Repo](https://github.com/funstory-ai/BabelDOC))**

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

*   **PDF Translation:** Translates scientific papers and other PDF documents.
*   **Online Service:**  Beta version available at [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) (1000 free pages per month).
*   **Self-Deployment:** Supports self-deployment using [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) with a WebUI.
*   **Command Line Interface (CLI):**  Provides a user-friendly CLI for direct translation tasks.
*   **Python API:** Offers a Python API for integration into other programs.
*   **Bilingual Output:**  Option to generate bilingual PDF files for side-by-side comparison.
*   **Offline Assets Management:**  Supports generating and restoring offline assets packages for environments without internet access or faster setup.
*   **Glossary Support:**  Allows integration of glossaries for customized translations.

## Getting Started

### Installation

We recommend using `uv` for environment management. Follow the [uv installation instructions](https://github.com/astral-sh/uv#installation) and set up your `PATH`.

**Install BabelDOC:**

```bash
uv tool install --python 3.12 BabelDOC
```

**Use the `babeldoc` command:**

```bash
babeldoc --help
```

**Example Translation:**

```bash
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
```

See the original README for details on installing from source and using the Python API.

## Advanced Options

Refer to the original README for the full list of advanced options, including language selection, PDF processing options, translation service configurations, OpenAI-specific settings, glossary options, and output control.

**Important Note:**  For end-users, we recommend using the [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) online service or self-hosting via [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).  The CLI is primarily for debugging.

## Preview

<div align="center">
<img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## We are hiring

See details: [EN](https://github.com/funstory-ai/jobs) | [ZH](https://github.com/funstory-ai/jobs/blob/main/README_ZH.md)

## Offline Assets Management

*   Generate an offline assets package: `babeldoc --generate-offline-assets /path/to/output/dir`
*   Restore from a package: `babeldoc --restore-offline-assets /path/to/offline_assets_*.zip` or from a directory.

## Configuration File

You can configure BabelDOC using a TOML file.  See the original README for an example.

## Contribution

Contributions are welcome! See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide and the [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).  Active contributors can receive Pro membership codes for Immersive Translation, see details at: [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md)

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