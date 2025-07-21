<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="marimo logo">
</p>

<p align="center">
  <em>A reactive Python notebook that's reproducible, git-friendly, and deployable as scripts or apps.</em>
</p>

<p align="center">
  <a href="https://docs.marimo.io" target="_blank"><strong>Docs</strong></a> ·
  <a href="https://marimo.io/discord?ref=readme" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://docs.marimo.io/examples/" target="_blank"><strong>Examples</strong></a> ·
  <a href="https://marimo.io/gallery/" target="_blank"><strong>Gallery</strong></a> ·
  <a href="https://www.youtube.com/@marimo-team/" target="_blank"><strong>YouTube</strong></a>
</p>

<p align="center">
  <b>English | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Chinese.md" target="_blank"><b>简体中文</b></a>
  <b> | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Japanese.md" target="_blank"><b>日本語</b></a>
  <b> | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Spanish.md" target="_blank"><b>Español</b></a>
</p>

<p align="center">
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI Version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda Version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads" alt="PyPI Downloads">
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" alt="Conda Downloads">
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

## Marimo: The Reactive Python Notebook for Data Science

**Marimo is a revolutionary Python notebook that provides a reactive, reproducible, and shareable environment for data science.** Unlike traditional notebooks, marimo ensures your code, outputs, and state are always consistent, making it perfect for research, communication, and application development.  Check out the original repo for more details: [https://github.com/marimo-team/marimo](https://github.com/marimo-team/marimo).

**Key Features:**

*   🚀 **Batteries-Included:** Replaces Jupyter, Streamlit, and more, offering a comprehensive data science environment.
*   ⚡️ **Reactive Execution:** Automatic updates of dependent cells when you modify code or interact with UI elements, ensuring consistency.
*   🖐️ **Interactive UI:** Easily bind sliders, tables, plots, and other UI elements to Python code without complex callbacks.
*   🐍 **Git-Friendly:** Notebooks are stored as standard `.py` files, making version control simple.
*   🛢️ **Data-Focused:** Built-in SQL support for querying data, filtering, and searching dataframes.
*   🤖 **AI-Enhanced:** Generate code and entire notebooks tailored for data work with AI assistants.
*   🔬 **Reproducible Results:** No hidden state, deterministic execution, and built-in package management.
*   🏃 **Executable Scripts:** Run notebooks as standard Python scripts with command-line arguments.
*   🛜 **Shareable:** Deploy notebooks as interactive web apps or slides, and even run them in the browser via WASM.
*   🧩 **Reusable Code:** Import functions and classes from other notebooks.
*   🧪 **Testable:** Integrates with pytest for easy testing of your notebooks.
*   ⌨️ **Modern Editor:** Includes features like GitHub Copilot integration, AI assistants, Vim keybindings, and a variable explorer.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

## Core Concepts

### A Reactive Programming Environment

Marimo offers a true reactive programming environment.  When a cell is run, marimo automatically updates all dependent cells, eliminating the need for manual re-running and ensuring your code and outputs are always in sync.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Execution GIF" />

### Advanced Features

*   **Compatible with expensive notebooks.** Configure marimo to mark cells as stale instead of automatically running them, preserving resources.
*   **Synchronized UI elements.** Interactive UI elements like sliders and dropdowns are automatically synchronized with the cells that use them.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="Interactive UI GIF" />

*   **Interactive dataframes.** Easily page through, search, filter, and sort millions of rows with no code required.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Interactive Dataframe GIF" />

*   **Generate cells with data-aware AI.** Utilize an AI assistant that's highly specialized for working with data to generate code based on the context of your variables.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI Code Generation GIF" />

*   **Query data with SQL.** Build SQL queries that depend on Python values and execute them against various data sources, returning results as Python dataframes.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Query Example" />

## Quickstart

_The [marimo concepts
playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq)
on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an
overview of many features._

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with additional dependencies that unlock SQL cells, AI completion, and more, run:

```bash
pip install marimo[recommended]
```

**Basic Commands:**

*   **Create/Edit Notebooks:** `marimo edit`
*   **Run as Web App:** `marimo run your_notebook.py`

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Marimo App Example" />

*   **Execute as Script:** `python your_notebook.py`
*   **Convert Jupyter Notebooks:** `marimo convert your_notebook.ipynb > your_notebook.py` or use our [web interface](https://marimo.io/convert)

**Tutorials:**
List all tutorials:

```bash
marimo tutorial --help
```

## Learn More

Get started and explore advanced features.

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Tutorial">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Inputs">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Plots">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Layout">
      </a>
    </td>
  </tr>
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html"> Tutorial </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html"> Inputs </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html"> Plots </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html"> Layout </a>
    </td>
  </tr>
  <tr>
    <td>
      <a target="_blank" href="https://marimo.app/l/c7h6pz">
        <img src="https://marimo.io/shield.svg" alt="Marimo Example">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Marimo Example">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Marimo Example">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Marimo Example">
      </a>
    </td>
  </tr>
</table>

Check out our [docs](https://docs.marimo.io), [usage examples](https://docs.marimo.io/examples/), and our [gallery](https://marimo.io/gallery) to learn more.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Questions? Reach out to us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join our community!

*   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.**

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS Logo" />

## Inspiration ✨

marimo's development was inspired by projects like Pluto.jl and ObservableHQ, aiming to provide a better programming environment for data science and communication.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="Marimo Logo">
</p>
```
Key improvements and SEO considerations:

*   **Clear Headline:** Uses "Marimo: The Reactive Python Notebook for Data Science" to immediately target relevant keywords.
*   **One-Sentence Hook:**  Provides a concise and engaging overview of the product's value.
*   **Keyword Optimization:**  Includes relevant keywords like "reactive Python notebook," "data science," "reproducible," "shareable," "interactive," and related terms throughout the text.
*   **Structured Format:** Uses headings, bullet points, and clear visuals to improve readability and SEO.
*   **Emphasis on Benefits:** Focuses on the benefits of using marimo (reproducibility, interactivity, ease of sharing, etc.) which is more appealing to users and search engines.
*   **Call to Action:** Encourages users to "Learn More" and provides clear links to key resources.
*   **Alt Text on Images:**  Added descriptive `alt` text to all images.
*   **Clear Installation Instructions:** Provides clear and concise installation instructions.
*   **Community Section:** Promotes community engagement with links to relevant platforms.
*   **NumFOCUS Mention:** Highlights affiliation with NumFOCUS, adding credibility.
*   **Inspiration Section:** Explains marimo's origins.