<!-- # BabelDOC: PDF Translation Library -->

## BabelDOC: Translate Scientific Papers with Ease

BabelDOC is your all-in-one solution for translating scientific PDF papers, offering both online and offline capabilities. **Effortlessly translate your research papers with BabelDOC – the ultimate tool for bridging language barriers in academia!** ([View on GitHub](https://github.com/funstory-ai/BabelDOC))

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

### Key Features:

*   **Scientific Paper Translation:** Focused on PDF scientific papers, providing accurate and context-aware translations.
*   **Online & Offline Options:** Utilize our online service, or self-deploy for offline use, giving you flexibility in how you translate.
*   **Command Line Interface:** Provides a user-friendly CLI for simple translation tasks.
*   **Python API:** Integrate BabelDOC's powerful translation capabilities directly into your Python projects.
*   **Bilingual Output:** Generate bilingual (original and translated) PDF files.
*   **Customization:**  Supports advanced options like language selection, OpenAI integration, glossary support, and output control for tailored translation.

### Get Started:

#### Online Service

Try our beta version: [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) (1000 free pages/month).

#### Self-Deployment

Deploy using [PDFMathTranslate 2.0](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

#### Command Line Interface

1.  **Install using [uv](https://github.com/astral-sh/uv) (Recommended):**
    ```bash
    uv tool install --python 3.12 BabelDOC
    babeldoc --help
    ```
    ```bash
    babeldoc --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"  --files example.pdf
    ```
2.  **Install from Source:**
    ```bash
    git clone https://github.com/funstory-ai/BabelDOC
    cd BabelDOC
    uv run babeldoc --help
    ```
    ```bash
    uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
    ```

### Advanced Options

*   **Language:** Specify source/target languages (`--lang-in`, `--lang-out`).
*   **PDF Processing:** Control page selection, splitting, cleaning, and dual-page layouts.
*   **Translation Service:** Choose OpenAI or other compatible APIs (`--openai`).
*   **Glossary Support:** Enhance translations using custom glossaries (`--glossary-files`).
*   **Output Control:** Manage output directory, debug settings, and report intervals.

See the complete list of options in the original [README](https://github.com/funstory-ai/BabelDOC).

### Python API

The current recommended way to call BabelDOC in Python is to call the `high_level.do_translate_async_stream` function of [pdf2zh next](https://github.com/PDFMathTranslate/PDFMathTranslate-next).

### We are Hiring

See details: [EN](https://github.com/funstory-ai/jobs) | [ZH](https://github.com/funstory-ai/jobs/blob/main/README_ZH.md)

### Roadmap and Contribution

Learn about our future goals and how you can contribute to BabelDOC by reading the [CONTRIBUTING](https://github.com/funstory-ai/yadt/blob/main/docs/CONTRIBUTING.md) guide.

### Star History

<a href="https://star-history.com/#funstory-ai/babeldoc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=funstory-ai/babeldoc&type=Date"/>
 </picture>
</a>