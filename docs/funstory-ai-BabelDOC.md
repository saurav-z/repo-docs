# BabelDOC: Translate Scientific PDFs with Ease

**Instantly translate scientific PDF papers, unlocking knowledge in multiple languages!**  [View on GitHub](https://github.com/funstory-ai/BabelDOC)

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-darkmode-with-transparent-background-IKuNO1.svg" width="320px" alt="BabelDOC"/>
  <img src="https://s.immersivetranslate.com/assets/uploads/babeldoc-big-logo-with-transparent-background-2xweBr.svg" width="320px" alt="BabelDOC"/>
</picture>

<!-- PyPI -->
<p>
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

*   **Accurate PDF Translation:** Translate scientific papers while preserving the original layout and formatting.
*   **Online & Self-Deployment Options:** Choose from an easy-to-use online service or self-deploy for greater control ([Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) & [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next)).
*   **Command Line Interface (CLI):**  Translate PDFs directly from your terminal.
*   **Python API:** Integrate BabelDOC functionality into your own applications.
*   **Dual-Language Output:** Generate bilingual PDFs for easy comparison.
*   **OpenAI Integration:** Leverage the power of OpenAI for high-quality translations.
*   **Glossary Support:** Improve translation accuracy by incorporating custom glossaries.
*   **Offline Asset Management:** Easily handle offline environments with assets packages.

## How to Get Started

### Install via PyPI

```bash
uv tool install --python 3.12 BabelDOC
babeldoc --help
```
```bash
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf

# multiple files
babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example1.pdf --files example2.pdf
```

### Install from Source

```bash
git clone https://github.com/funstory-ai/BabelDOC
cd BabelDOC
uv run babeldoc --help
```
```bash
uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
# multiple files
uv run babeldoc --files example.pdf --files example2.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
```

##  Preview

<div align="center">
<img src="https://s.immersivetranslate.com/assets/r2-uploads/images/babeldoc-preview.png" width="80%"/>
</div>

## Advanced Options

For detailed options and configurations, refer to the [Advanced Options](#advanced-options) section in the original README.  This includes options for language, PDF processing, translation services (including OpenAI configuration), output control, glossary management, and offline asset management.

**Key areas to explore:**

*   **Language Options:** `--lang-in`, `--lang-out`
*   **PDF Processing Options:**  `--files`, `--pages`, `--enhance-compatibility`, `--max-pages-per-part`
*   **Translation Service Options:** `--openai`, `--openai-model`, `--openai-api-key`
*   **Glossary Options:** `--glossary-files`
*   **Offline Assets:** `--generate-offline-assets`, `--restore-offline-assets`

## Integration with Zotero

*   Immersive Translate Pro members can use the [immersive-translate/zotero-immersivetranslate](https://github.com/immersive-translate/zotero-immersivetranslate) plugin
*   PDFMathTranslate self-deployed users can use the [guaguastandup/zotero-pdf2zh](https://github.com/guaguastandup/zotero-pdf2zh) plugin

## We are Hiring

See details: [EN](https://github.com/funstory-ai/jobs) | [ZH](https://github.com/funstory-ai/jobs/blob/main/README_ZH.md)

## Background

This project is designed to provide a standard pipeline and interface for the translation of PDFs, as an alternative to manual translation, and competing projects like Mathpix. BabelDOC offers an intermediate representation of the parsed results from a PDF and renders them into a new PDF or other format.

## Roadmap

*   [ ] Add line support
*   [ ] Add table support
*   [ ] Add cross-page/cross-column paragraph support
*   [ ] More advanced typesetting features
*   [ ] Outline support
*   [ ] ...

## Versioning

BabelDOC follows [Semantic Versioning](https://semver.org/) and [Pride Versioning](https://pridever.org/) "0.MAJOR.MINOR".

## Known Issues

*   Parsing errors in the author and reference sections; they get merged into one paragraph after translation.
*   Lines are not supported.
*   Does not support drop caps.
*   Large pages will be skipped.

## How to Contribute

Contributions are welcome!  See the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide. Adhere to the [Code of Conduct](https://github.com/funstory-ai/yadt/blob/main/docs/CODE_OF_CONDUCT.md).

Active contributors can earn [Immersive Translation](https://immersivetranslate.com) Pro membership redemption codes. See [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md).

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