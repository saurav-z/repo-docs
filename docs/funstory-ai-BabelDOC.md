<!-- # BabelDOC: Your PDF Translation Solution -->

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
    <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
  </picture>

  <p>
    <a href="https://pypi.org/project/BabelDOC/">
      <img src="https://img.shields.io/pypi/v/BabelDOC" alt="PyPI version">
    </a>
    <a href="https://pepy.tech/projects/BabelDOC">
      <img src="https://static.pepy.tech/badge/BabelDOC" alt="Downloads">
    </a>
    <a href="./LICENSE">
      <img src="https://img.shields.io/github/license/funstory-ai/BabelDOC" alt="License">
    </a>
    <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
      <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white" alt="Telegram">
    </a>
  </p>

  <a href="https://trendshift.io/repositories/13358" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/13358" alt="funstory-ai%2FBabelDOC | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

## BabelDOC: Effortlessly Translate PDF Documents

BabelDOC is a powerful library for translating PDF scientific papers, providing both online and self-deployment options.  [Explore the BabelDOC project on GitHub](https://github.com/funstory-ai/BabelDOC).

**Key Features:**

*   **Online Service:**  Try the beta version on [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) with 1000 free pages per month.
*   **Self-Deployment:**  Leverage [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next) for self-hosting and a WebUI with various translation services.
*   **Command Line Interface (CLI):**  Easily translate documents with the included CLI.
*   **Python API:** Integrate BabelDOC into your own applications with the provided Python API.
*   **Bilingual PDF Generation:** Create side-by-side or dual-page PDFs for easy comparison.
*   **Offline Assets Management:** Generate and restore offline assets for environments without internet access.

## Getting Started

BabelDOC offers installation via PyPI and from source. We recommend using [uv](https://github.com/astral-sh/uv) for managing your virtual environment.

### Install from PyPI

1.  Follow the [uv installation instructions](https://github.com/astral-sh/uv#installation) and configure your `PATH` environment variable.
2.  Install BabelDOC using `uv tool install --python 3.12 BabelDOC`.
3.  Run `babeldoc --help` to verify the installation.

### Install from Source

1.  Clone the repository: `git clone https://github.com/funstory-ai/BabelDOC`
2.  Navigate to the project directory: `cd BabelDOC`
3.  Install dependencies and run `babeldoc`: `uv run babeldoc --help`

## Advanced Options

> [!NOTE]
> The CLI is primarily for debugging. For end-users, we recommend the [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) online service (1000 free pages/month) or the self-hosted [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

### Language Options

*   `--lang-in`, `-li`: Source language code (default: `en`).
*   `--lang-out`, `-lo`: Target language code (default: `zh`).

### PDF Processing Options

*   `--files`:  Specify input PDF document(s).
*   `--pages`, `-p`:  Define specific pages to translate (e.g., "1,2,1-,-3,3-5").
*   (And many more options - see original README for full details)

### Translation Service Options

*   `--qps`: QPS (Queries Per Second) limit for translation service (default: 4).
*   `--openai`: Enable OpenAI translation (default: `False`).
*   (And many more options - see original README for full details)

### OpenAI Specific Options

*   `--openai-model`: OpenAI model to use (default: `gpt-4o-mini`).
*   `--openai-base-url`: Base URL for OpenAI API.
*   `--openai-api-key`: API key for OpenAI service.

### Glossary Options

*   `--glossary-files`:  Comma-separated paths to glossary CSV files.

### Output Control

*   `--output`, `-o`: Output directory for translated files.
*   `--debug`: Enable debug logging.

### General Options

*   `--warmup`: Only download/verify assets and exit (default: `False`).

### Offline Assets Management

*   `--generate-offline-assets`:  Generate an offline assets package.
*   `--restore-offline-assets`:  Restore an offline assets package.

## Python API

> [!TIP]
> Use the Python API from [pdf2zh 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next/blob/main/pdf2zh_next/high_level.py) for full compatibility.

Refer to `babeldoc/main.py` for an example of using the BabelDOC Python API.

*   Ensure to call `babeldoc.format.pdf.high_level.init()` before using the API.
*   Validate input parameters for use.

## Background

BabelDOC builds upon previous projects in the document parsing and translation space, offering a comprehensive solution for scientific paper translation.

## Roadmap

*   Add line support
*   Add table support
*   Add cross-page/cross-column paragraph support
*   More advanced typesetting features
*   Outline support
*   ...

## Versioning

This project follows Semantic Versioning with "Pride Versioning." Version format: "0.MAJOR.MINOR."

## Known Issues

*   Parsing errors in the author and reference sections.
*   Line support is missing.
*   Drop caps are not supported.
*   Large pages may be skipped.

## How to Contribute

Contribute to YADT through our [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide. Please adhere to our [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).

Active contributors can receive monthly Pro membership redemption codes for Immersive Translation.  See [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md) for details.

## Acknowledgements

*   (List of acknowledgements)

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

> (See original for complete details)