<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg">
</p>

<p align="center">
  <em>A reactive Python notebook that's reproducible, git-friendly, and deployable as scripts or apps.</em>

<p align="center">
  <a href="https://docs.marimo.io" target="_blank"><strong>Docs</strong></a> ·
  <a href="https://marimo.io/discord?ref=readme" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://docs.marimo.io/examples/" target="_blank"><strong>Examples</strong></a> ·
  <a href="https://marimo.io/gallery/" target="_blank"><strong>Gallery</strong></a> ·
  <a href="https://www.youtube.com/@marimo-team/" target="_blank"><strong>YouTube</strong></a>
</p>

<p align="center">
  <b>English | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Traditional_Chinese.md" target="_blank"><b>繁體中文</b></a>
  <b> | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Chinese.md" target="_blank"><b>简体中文</b></a>
  <b> | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Japanese.md" target="_blank"><b>日本語</b></a>
  <b> | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Spanish.md" target="_blank"><b>Español</b></a>
</p>

<p align="center">
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi"/></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg"/></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="discord" /></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" /></a>
</p>

# marimo: The Reactive Python Notebook for Data Science 

marimo is a revolutionary reactive Python notebook that transforms the way you work with data. **[Explore the future of data science with marimo, where your code reacts in real-time!](https://github.com/marimo-team/marimo)**

## Key Features

*   🚀 **Batteries-included:** Replaces Jupyter, Streamlit, and more, streamlining your workflow.
*   ⚡️ **Reactive:** Automatic updates: Run a cell, and dependent cells update instantly.
*   🖐️ **Interactive:** Bind sliders, tables, and plots with ease. No callbacks needed.
*   🐍 **Git-Friendly:** Notebooks are stored as `.py` files for easy version control.
*   🛢️ **Data-Focused:** Work with dataframes and databases using SQL.
*   🤖 **AI-Native:** Generate cells with AI tailored for data work
*   🔬 **Reproducible:** Ensure consistent results with no hidden state and deterministic execution.
*   🏃 **Executable:** Run notebooks as Python scripts, customizable with CLI arguments.
*   🛜 **Shareable:** Deploy notebooks as interactive web apps or slides.
*   🧩 **Reusable:** Import functions and classes between notebooks.
*   🧪 **Testable:** Run pytest directly on your notebooks.
*   ⌨️ **Modern Editor:** Benefit from GitHub Copilot, AI assistants, and a powerful editor.

## Quickstart

Install marimo with:

```bash
pip install marimo[recommended]
```
or
```bash
pip install marimo && marimo tutorial intro
```

Then, create and edit notebooks using:

```bash
marimo edit
```

## Dive Deeper

marimo offers an array of features to elevate your data science projects:

### Reactive Environment

marimo ensures that your notebook code, outputs, and program state are consistent.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" />

### Compatible with Expensive Notebooks

Configure the runtime to be lazy, marking affected cells as stale instead of running them automatically.

### Interactive UI Elements

Interact with UI elements like sliders, dropdowns, and dataframe transformers, and the cells that use them are automatically re-run with their latest values.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" />

### Interactive Dataframes

Page, search, filter, and sort millions of rows without writing any code.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" />

### AI-Powered Code Generation

Generate code with an AI assistant highly specialized for working with data and get context about your variables in memory.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" />

### SQL Integration

Build SQL queries that depend on Python values and execute them against dataframes, databases, lakehouses, CSVs, Google Sheets, or anything else using our built-in SQL engine, which returns the result as a Python dataframe.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" />

### Dynamic Markdown

Create dynamic markdown parametrized by Python variables to tell dynamic stories that depend on Python data.

### Package Management

Install packages on import, serialize package requirements, and auto install them in isolated venv sandboxes.

### Deterministic Execution

Notebooks are executed in a deterministic order, based on variable references instead of cells' positions on the page.

### Performant Runtime

marimo runs only those cells that need to be run by statically analyzing your code.

### Batteries-Included

marimo comes with GitHub Copilot, AI assistants, Ruff code formatting, HTML export, fast code completion, a [VS Code
extension](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo),
an interactive dataframe viewer, and [many more](https://docs.marimo.io/guides/editor_features/index.html)
quality-of-life features.

## Usage

*   **Create & Edit:** `marimo edit`
*   **Run as App:** `marimo run your_notebook.py`
*   **Execute as Script:** `python your_notebook.py`
*   **Convert Jupyter Notebooks:** `marimo convert your_notebook.ipynb > your_notebook.py` or via our [web interface](https://marimo.io/convert).

## Tutorials

List all tutorials:

```bash
marimo tutorial --help
```

## Learn More

Explore our [docs](https://docs.marimo.io), [usage examples](https://docs.marimo.io/examples/), and our [gallery](https://marimo.io/gallery) to get started.

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" />
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
        <img src="https://marimo.io/shield.svg"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg"/>
      </a>
    </td>
  </tr>
</table>

## Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Questions? Join us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join the marimo community:

*   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.** marimo is part of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" />

## Inspiration ✨

marimo is a reinvention of the Python notebook as a reproducible, interactive, and shareable Python program.

We are inspired by [Pluto.jl](https://github.com/fonsp/Pluto.jl), [ObservableHQ](https://observablehq.com/tutorials), and [Bret Victor's essays](http://worrydream.com/).

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px">
</p>
```
Key improvements and explanations:

*   **SEO Optimization:**  Incorporated relevant keywords like "Python notebook," "reactive," "data science," and "interactive" in the title and throughout the document.
*   **Clear Heading Structure:** Uses clear, descriptive headings (e.g., "Key Features," "Quickstart," "Dive Deeper," "Learn More," "Contributing") for improved readability and SEO.
*   **Concise Summary:**  The opening sentence grabs attention and quickly explains what marimo is.
*   **Bulleted Key Features:** Uses bullet points to highlight the core functionalities, making it easy to scan and understand the value proposition.
*   **Concise Explanations:** Each feature is accompanied by a brief, clear explanation.
*   **Emphasis on Benefits:** The descriptions focus on the benefits of using marimo (e.g., "streamlining your workflow," "ensuring consistent results").
*   **Call to Action:** Encourages users to try marimo, learn more, and contribute.
*   **Links to Key Resources:** Prominently features links to documentation, examples, and the Discord community.
*   **Removed Redundancy:** Streamlined some of the repetitive language.
*   **Code Blocks with Examples:** Provides concise code snippets for installation and usage.
*   **Clean Formatting:** The Markdown is well-formatted for easy readability.
*   **Contextual Images:** Images are kept that are relevant to the content and add visual interest.
*   **Improved Readability** The flow and language is improved to make it easier to digest the information.