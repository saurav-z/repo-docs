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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI version"/></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda version"/></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord server" /></a>
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"/></a>
</p>

## marimo: The Reactive Python Notebook for Data Science

**marimo** is a modern, reactive Python notebook designed for data scientists, offering reproducibility, interactive features, and effortless deployment.  [Get started with marimo on GitHub](https://github.com/marimo-team/marimo).

**Key Features:**

*   🚀 **Batteries-Included:** Replaces Jupyter, Streamlit, and more, offering a comprehensive data science environment.
*   ⚡️ **Reactive Execution:** Automatically updates dependent cells when a cell's input changes, ensuring code and outputs remain consistent.
*   🖐️ **Interactive UIs:** Easily create interactive elements like sliders, tables, and plots, all linked directly to your Python code, without callbacks.
*   🐍 **Git-Friendly:** Notebooks are stored as standard `.py` files, simplifying version control and collaboration.
*   🛢️ **Data-Focused:** Seamlessly work with data using SQL, filter and search dataframes, and connect to databases.
*   🤖 **AI-Enhanced:** Generate Python code tailored to data tasks with AI assistance, including zero-shot notebook generation.
*   🔬 **Reproducible:** Deterministic execution and built-in package management ensure consistent results.
*   🏃 **Executable Scripts:** Execute notebooks as Python scripts, with the ability to parameterize via CLI arguments.
*   🛜 **Shareable & Deployable:** Easily deploy notebooks as interactive web apps or slides, even run them in the browser using WASM.
*   🧩 **Reusable Code:** Import functions and classes between notebooks for modular code design.
*   🧪 **Testable Notebooks:** Integrate pytest for comprehensive testing of your notebooks.
*   ⌨️ **Modern Editor:** Benefit from a modern editor with features like GitHub Copilot integration, AI assistants, Vim keybindings, and a variable explorer.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [quickstart](#quickstart) for a primer on our CLI._

## Core Concepts

marimo's reactive design solves many problems associated with traditional notebooks like Jupyter.

**Reactive Environment:** Run a cell and marimo *reacts* by automatically running dependent cells, eliminating manual re-running and the risk of errors. Deleting a cell cleans up the state, preventing hidden state issues.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Notebooks">

**Compatible with Expensive Notebooks:** Configure marimo to be lazy, marking cells as "stale" instead of automatically running them. This preserves program state while preventing unintended execution of computationally expensive cells.

**Synchronized UI Elements:** Interact with sliders, dropdowns, and dataframe transformers, and the cells that use them are automatically re-run with their latest values.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI elements">

**Interactive DataFrames:**  Page through, search, filter, and sort millions of rows blazingly fast, directly in the notebook.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Interactive DataFrames">

**AI-Assisted Code Generation:** Generate code with AI assistance specialized for data work, with context from your variables, including zero-shot entire notebooks.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI-assisted generation">

**SQL Integration:** Build SQL queries that depend on Python values and execute them against dataframes, databases, lakehouses, CSVs, or Google Sheets using our built-in SQL engine.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL integration">

Your notebooks are still pure Python, even if they use SQL.

**Dynamic Markdown:** Use markdown parametrized by Python variables to tell dynamic stories that depend on Python data.

**Built-in Package Management:**  Install packages on import and even serialize package requirements directly in notebook files, with auto-installation in isolated venvs.

**Deterministic Execution Order:** Notebooks are executed in a deterministic order, based on variable references.

**Performant Runtime:**  marimo runs only those cells that need to be run by statically analyzing your code.

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

To install with additional dependencies that unlock SQL cells, AI completion, and more,
run

```bash
pip install marimo[recommended]
```

**Create & Edit Notebooks:**

```bash
marimo edit
```

**Run as Web App:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Run as web app">

**Execute as Script:**

```bash
python your_notebook.py
```

**Convert Jupyter Notebooks:**

```bash
marimo convert your_notebook.ipynb > your_notebook.py
```
or use our [web interface](https://marimo.io/convert).

**Tutorials:**

```bash
marimo tutorial --help
```

## Get Help

*   [FAQ](https://docs.marimo.io/faq.html)
*   [Discord](https://marimo.io/discord?ref=readme)

## Learn More

Explore marimo's capabilities with these resources:

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
        <img src="https://marimo.io/shield.svg" alt="Example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Example"/>
      </a>
    </td>
  </tr>
</table>

*   [Documentation](https://docs.marimo.io)
*   [Examples](https://docs.marimo.io/examples/)
*   [Gallery](https://marimo.io/gallery)

## Contributing

We welcome all contributions!  Review [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Questions? Reach out to us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join our community!

*   🌟 [Star on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Discord](https://marimo.io/discord?ref=readme)
*   📧 [Newsletter](https://marimo.io/newsletter)
*   ☁️ [Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [GitHub Discussions](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Twitter](https://twitter.com/marimo_io)
*   🎥 [YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS Affiliated Project:** marimo is part of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFocus">

## Inspiration ✨

marimo is a **reinvention** of the Python notebook, designed for reproducible and shareable Python programs.  We're inspired by projects like [Pluto.jl](https://github.com/fonsp/Pluto.jl) and [ObservableHQ](https://observablehq.com/tutorials), with influences from Bret Victor's work on dataflow programming.