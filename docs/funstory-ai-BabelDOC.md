# BabelDOC: Translate PDF Scientific Papers with Ease

**BabelDOC** is a powerful library for translating PDF scientific papers, offering both online and self-hosted options to revolutionize your research workflow. [Visit the original repository](https://github.com/funstory-ai/BabelDOC) for more details.

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

*   **Accurate PDF Translation:** Translate scientific papers with high precision.
*   **Online Service:** Beta version available at [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) with 1000 free pages per month.
*   **Self-Deployment:** Leverage [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting.
*   **Command-Line Interface:** Convenient CLI for direct translation tasks.
*   **Python API:** Integrate translation into your Python projects.
*   **Bilingual Comparison:** Generate side-by-side original and translated PDF outputs.
*   **OpenAI Integration:** Supports OpenAI models for advanced translation.
*   **Offline Asset Management:** Generate and restore offline asset packages for environments without internet access.

## Getting Started

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) to install BabelDOC.

1.  Install `uv` and set up the `PATH` environment variable. See [uv installation](https://github.com/astral-sh/uv#installation)
2.  Use the following command to install:

```bash
uv tool install --python 3.12 BabelDOC

babeldoc --help
```

### Usage

Translate a PDF using the command line:

```bash
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
```

Or, for multiple files:

```bash
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
```

### Install from Source

1.  Clone the repository:

```bash
git clone https://github.com/funstory-ai/BabelDOC
```

2.  Navigate to the project directory:

```bash
cd BabelDOC
```

3.  Install dependencies and run babeldoc:

```bash
uv run babeldoc --help
```

## Advanced Options

For detailed options, refer to the [original README](https://github.com/funstory-ai/BabelDOC).

### Supported Languages
[Supported Language](https://funstory-ai.github.io/BabelDOC/supported_languages/)

### Preview

<div align="center">
<img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## We are hiring

See details: [EN](https://github.com/funstory-ai/jobs) | [ZH](https://github.com/funstory-ai/jobs/blob/main/README_ZH.md)

## Contribution

Contributions are welcome! See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide for details.

## Acknowledgements

-   [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
-   [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
-   [pdfminer](https://github.com/pdfminer/pdfminer.six)
-   [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
-   [Asynchronize](https://github.com/multimeric/Asynchronize/tree/master?tab=readme-ov-file)
-   [PriorityThreadPoolExecutor](https://github.com/oleglpts/PriorityThreadPoolExecutor)

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