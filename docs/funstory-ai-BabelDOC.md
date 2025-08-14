# BabelDOC: Effortlessly Translate Scientific Papers and Documents

**Quickly translate PDF scientific papers, research documents, and more!** BabelDOC offers powerful PDF translation capabilities, providing both online and self-hosted options. Learn more at the [original repo](https://github.com/funstory-ai/BabelDOC).

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

## Key Features

*   **High-Quality Translation:** Translate scientific papers and documents.
*   **Online and Self-Hosted Options:** Utilize the [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) beta, or self-deploy with [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).
*   **Command Line Interface (CLI):** Easy-to-use CLI for quick translations.
*   **Python API:** Integrate BabelDOC into your Python projects.
*   **Bilingual PDF Output:** Generate translated PDFs with original and translated text side-by-side or alternating pages.
*   **Extensive Configuration:** Fine-tune your translations with various options, including language selection, OpenAI integration, and glossary support.
*   **Offline Assets Management:** Generate and restore offline asset packages for use in air-gapped or offline environments.

## Getting Started

BabelDOC offers flexible installation options using `uv`.

### Install from PyPI

1.  Follow the [uv installation instructions](https://github.com/astral-sh/uv#installation) to install uv and set up the PATH environment variable.
2.  Install BabelDOC using:

    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```

3.  Use the `babeldoc` command, for example:

    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf

    # multiple files
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
    ```

### Install from Source

1.  Follow the [uv installation instructions](https://github.com/astral-sh/uv#installation) to install uv and set up the PATH environment variable.
2.  Clone the repository:

    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    ```

3.  Install dependencies and run BabelDOC:

    ```bash
    uv run babeldoc --help
    ```

4.  Use the `uv run babeldoc` command:

    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"

    # multiple files
    uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

### Advanced Options

>   [!NOTE]
>   For end-users, the recommended method is to use the **Online Service** (beta) or **Self-deployment** options. The CLI is primarily for debugging.

BabelDOC provides a range of options to customize your translations.

*   **Language Options:** Specify input and output languages (`--lang-in`, `--lang-out`).
*   **PDF Processing Options:** Control page selection, text cleaning, dual PDF output, and compatibility enhancements.
*   **Translation Service Options:** Configure QPS limits, ignore cache, enable OpenAI, and set custom prompts.
*   **OpenAI Specific Options:** Configure OpenAI model, base URL, and API key.
*   **Glossary Options:** Provide custom glossaries for accurate term translation.
*   **Output Control:** Set the output directory, debug level, and report interval.
*   **Offline Assets Management:** Generate and restore offline assets packages.
*   **Configuration File:** Use TOML configuration files for easy setup.

[See full details on Advanced Options](#advanced-options)

### Supported Languages

[See supported languages](https://funstory-ai.github.io/BabelDOC/supported_languages/)

## Preview

<div align="center">
<img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## We are hiring

See details: [EN](https://github.com/funstory-ai/jobs) | [ZH](https://github.com/funstory-ai/jobs/blob/main/README_ZH.md)

## Python API

>   [!TIP]
>   Before pdf2zh 2.0 is released, you can temporarily use BabelDOC's Python API. However, after pdf2zh 2.0 is released, please directly use pdf2zh's Python API.
>   This project's Python API does not guarantee any compatibility. However, the Python API from pdf2zh will guarantee a certain level of compatibility.
>   We do not provide any technical support for the BabelDOC API.
>   When performing secondary development, please refer to [pdf2zh 2.0 high level](https://github.com/PDFMathTranslate/PDFMathTranslate-next/blob/main/pdf2zh_next/high_level.py) and ensure that BabelDOC runs in a subprocess.

You can refer to the example in [main.py](https://github.com/funstory-ai/yadt/blob/main/babeldoc/main.py) to use BabelDOC's Python API.

Please note:

1.  Make sure call `babeldoc.format.pdf.high_level.init()` before using the API
2.  The current `TranslationConfig` does not fully validate input parameters, so you need to ensure the validity of input parameters
3.  For offline assets management, you can use the following functions:

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

    >   [!TIP]
    >
    >   1. The offline assets package name cannot be modified because the file list hash is encoded in the name.
    >   2. When using in production environments, it's recommended to pre-generate the assets package and include it with your application distribution.
    >   3. The package verification ensures that all required assets are intact and match their expected checksums.

## Background

This project aims to standardize and simplify the PDF translation process, leveraging advancements in PDF parsing and rendering techniques.  It draws inspiration from projects like Mathpix, Doc2X, minerU, and PDFMathTranslate, and addresses challenges in parsing, rendering, and maintaining the original document structure during translation.

## Roadmap

*   Add line support
*   Add table support
*   Add cross-page/cross-column paragraph support
*   More advanced typesetting features
*   Outline support
*   ...

The initial 1.0 version focuses on translating documents from English to Simplified Chinese, Traditional Chinese, Japanese, and Spanish while maintaining layout and content accuracy.

## Version Number Explanation

This project uses a combination of [Semantic Versioning](https://semver.org/) and [Pride Versioning](https://pridever.org/). The version number format is: "0.MAJOR.MINOR".

-   MAJOR: Incremented by 1 when API incompatible changes are made or when proud improvements are implemented.

-   MINOR: Incremented by 1 when any API compatible changes are made.

## Known Issues

*   Parsing errors in author and reference sections.
*   Line support is not yet implemented.
*   Does not support drop caps.
*   Large pages may be skipped.

## How to Contribute

We welcome contributions!  Please see the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide for more information. The [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md) applies.

[Immersive Translation](https://immersivetranslate.com) sponsors monthly Pro membership redemption codes for active contributors, see details at: [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md)

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

>   [!WARNING]
>   **Important Interaction Note for `--auto-enable-ocr-workaround`:**
>
>   When `--auto-enable-ocr-workaround` is set to `true` (either via command line or config file):
>
>   1.  During the initial setup, the values for `ocr_workaround` and `skip_scanned_detection` will be forced to `false` by `TranslationConfig`, regardless of whether you also set `--ocr-workaround` or `--skip-scanned-detection` flags.
>   2.  Then, during the scanned document detection phase (`DetectScannedFile` stage):
>       *   If the document is identified as heavily scanned (e.g., >80% scanned pages) AND `auto_enable_ocr_workaround` is `true` (i.e., `translation_config.auto_enable_ocr_workaround` is true), the system will then attempt to set both `ocr_workaround` to `true` and `skip_scanned_detection` to `true`.
>
>   This means that `--auto-enable-ocr-workaround` effectively gives the system control to enable OCR processing for scanned documents, potentially overriding manual settings for `--ocr-workaround` and `--skip_scanned_detection` based on its detection results. If the document is *not* detected as heavily scanned, then the initial `false` values for `ocr_workaround` and `skip_scanned_detection` (forced by `--auto-enable-ocr-workaround` at the `TranslationConfig` initialization stage) will remain in effect unless changed by other logic.