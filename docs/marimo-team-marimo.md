<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="Marimo Logo">
</p>

<p align="center">
  <em>Revolutionize your Python workflow with marimo, a reactive notebook that's reproducible, git-friendly, and deployable as scripts or apps.</em>
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
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Traditional_Chinese.md" target="_blank"><b>繁體中文</b></a>
  <b> | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Chinese.md" target="_blank"><b>简体中文</b></a>
  <b> | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Japanese.md" target="_blank"><b>日本語</b></a>
  <b> | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Spanish.md" target="_blank"><b>Español</b></a>
</p>

<p align="center">
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI version"/></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda version"/></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord server" /></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"/></a>
</p>

## What is marimo?

**marimo is a next-generation Python notebook designed for data science and research, offering a reactive, reproducible, and shareable environment.**  It replaces traditional notebooks like Jupyter and provides a more streamlined and powerful way to work with Python.

**Key Features:**

*   🚀 **Batteries-Included:** Replaces multiple tools like Jupyter, Streamlit, and ipywidgets.
*   ⚡️ **Reactive Execution:** Automatically updates dependent cells when you change code or interact with UI elements.  Eliminates manual cell re-running.
*   🖐️ **Interactive UI:**  Easily create interactive elements like sliders, tables, and plots with no callback functions needed.
*   🐍 **Git-Friendly:** Notebooks are stored as plain `.py` files for easy version control.
*   🛢️ **Data-Focused:**  First-class support for SQL queries, dataframes, and database interactions, built for the modern data professional.
*   🤖 **AI-Powered:** Integrated AI features to generate code with the assistance of AI, tailored for data tasks.
*   🔬 **Reproducible:**  Guarantees consistent code execution and no hidden state. Deterministic execution order. Includes built-in package management.
*   🏃 **Executable as Scripts:** Run notebooks as standard Python scripts, with CLI argument support.
*   🛜 **Shareable:** Deploy interactive web apps or slides directly from your notebooks. Run in the browser via WASM.
*   🧩 **Reusable Code:** Import functions and classes from one notebook to another.
*   🧪 **Testable:** Run pytest on your marimo notebooks.
*   ⌨️ **Modern Editor:** Built-in support for GitHub Copilot, AI assistants, Vim keybindings, and more.

```python
pip install marimo && marimo tutorial intro
```
_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

<a name="quickstart"></a>

## Getting Started

### Installation
```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```
For SQL cells, AI features and more:
```bash
pip install marimo[recommended]
```

### Core Commands

*   **Create/Edit Notebooks:** `marimo edit`
*   **Run as Web App:** `marimo run your_notebook.py`
*   **Execute as Script:** `python your_notebook.py`
*   **Convert Jupyter Notebooks:** `marimo convert your_notebook.ipynb > your_notebook.py` (or use the [web interface](https://marimo.io/convert))
*   **List Tutorials:** `marimo tutorial --help`

### Quickstart

The [marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq) on our [YouTube channel](https://www.youtube.com/@marimo-team) provides an overview of the features.

## Deep Dive into Features

### Reactive Programming Environment

marimo ensures consistency by automatically updating related parts of your notebook.  This is a vast improvement over standard notebooks.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Execution Example" />

### Compatible with Expensive Notebooks

Configure the runtime to be lazy and mark cells as stale instead of running them.

### Synchronized UI Elements

Easily integrate UI elements, such as sliders and dropdowns, that update in real-time.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI Elements Example" />

### Interactive DataFrames

Quickly page, search, filter, and sort through large datasets directly within your notebook.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Dataframe Example" />

### AI-Powered Code Generation

Generate data-aware code with specialized AI assistants that have the context of your notebook variables.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI Code Generation Example" />

### SQL Integration

Build SQL queries that depend on Python values and execute them against dataframes, databases, lakehouses, CSVs, Google Sheets, or anything else using our built-in SQL engine, which returns the result as a Python dataframe.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Cell Example" />

### Dynamic Markdown

Create dynamic markdown that changes with Python variables.

### Built-in Package Management

Install packages directly from within the notebook and manage dependencies easily.

### Deterministic Execution

Notebooks run in a predictable order based on variable dependencies.

### Performant Runtime

marimo runs only those cells that need to be run by statically analyzing your code.

### Extensive Editor Features

marimo includes features like GitHub Copilot, AI assistants, Ruff code formatting, fast code completion, and more.

## Further Exploration

For a deeper dive, explore the following resources:

*   [Documentation](https://docs.marimo.io)
*   [Usage Examples](https://docs.marimo.io/examples/)
*   [Gallery](https://marimo.io/gallery)

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Tutorial"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Inputs"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Plots"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Layout"/>
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
        <img src="https://marimo.io/shield.svg" alt="Playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Example Notebook"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Example Notebook"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Example Notebook"/>
      </a>
    </td>
  </tr>
</table>

## Contributing

[Contribute](https://github.com/marimo-team/marimo) to the project.

## Community & Support

*   [Join our Discord](https://marimo.io/discord?ref=readme) for help and discussions.
*   [Star us on GitHub](https://github.com/marimo-team/marimo)
*   [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   [Join our Cloud Waitlist](https://marimo.io/cloud)
*   [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   [Follow us on Twitter](https://twitter.com/marimo_io)
*   [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**marimo is a NumFOCUS affiliated project.**

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS Logo" />

## Inspiration

marimo reimaginies the Python notebook experience by providing a reproducible, interactive, and shareable Python program, built with better tools in mind.

**[Visit the GitHub repository](https://github.com/marimo-team/marimo) to learn more!**