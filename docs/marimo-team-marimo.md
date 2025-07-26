<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg">
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

# Marimo: The Reactive Python Notebook for Data Science and App Development

**Marimo is a next-generation Python notebook that revolutionizes data science workflows by offering reactivity, reproducibility, and seamless deployment.** Visit the [original repository](https://github.com/marimo-team/marimo) to learn more.

## Key Features

*   🚀 **Batteries-Included:** Replaces `jupyter`, `streamlit`, `jupytext`, `ipywidgets`, `papermill`, and more.
*   ⚡️ **Reactive Execution:**  Changes in a cell automatically trigger updates in dependent cells, ensuring code consistency.
*   🖐️ **Interactive:** Create interactive dashboards and applications by binding UI elements (sliders, tables, plots, and more) to Python code without the need for callbacks.
*   🐍 **Git-Friendly:** Notebooks are stored as `.py` files, facilitating version control and collaboration.
*   🛢️ **Data-Focused:**  Offers powerful SQL support, enabling direct querying of dataframes, databases, and data warehouses. Plus, filter and search dataframes.
*   🤖 **AI-Native:** Generate and explore with AI to make code and exploration more efficient.
*   🔬 **Reproducible:** Guarantees reproducible results with no hidden state and deterministic execution.
*   🏃 **Executable as Scripts:** Run notebooks as standard Python scripts, parameterized by command-line arguments.
*   🛜 **Shareable and Deployable:** Easily deploy notebooks as interactive web apps or slides, even in the browser with WASM.
*   🧩 **Reusable Components:** Import functions and classes between notebooks for modularity.
*   🧪 **Testable:** Integrate unit testing using `pytest`.
*   ⌨️ **Modern Editor:** Features a modern code editor with GitHub Copilot, AI assistants, Vim keybindings, a variable explorer, and more.

## Installation and Quickstart

Install marimo:

```bash
pip install marimo[recommended]
```

Then, explore a quick tutorial:

```bash
marimo tutorial intro
```

## Core Concepts

### Reactive Programming Environment

marimo ensures consistency in your code, outputs, and program state.  Run a cell, and marimo automatically updates the dependent cells, making manual re-runs a thing of the past.

### Interactive Dataframes and UI Elements

Interact with sliders, dropdowns, and other UI elements, and marimo will automatically rerun the cells that use them.

### AI-Powered Code Generation

Generate code with AI, specially tailored for data work, with context about your variables in memory; zero-shot entire notebooks.

### SQL Integration

Build SQL queries that depend on Python values and execute them against dataframes, databases, lakehouses, CSVs, Google Sheets, or anything else using our built-in SQL engine, which returns the result as a Python dataframe.

### Key Features

*   **Reproducible Notebooks**:  Deterministic execution order and built-in package management.
*   **Performance**: Runs only cells that need to be run.
*   **Extensive Functionality**: Includes GitHub Copilot, AI assistants, Ruff code formatting, HTML export, an interactive dataframe viewer, and much more.

## Learn More

Explore the full capabilities of marimo:

*   [Docs](https://docs.marimo.io)
*   [Examples](https://docs.marimo.io/examples/)
*   [Gallery](https://marimo.io/gallery)

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

Contributions are welcome! Please review [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> For questions or support, reach out [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join the marimo community:

*   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.** marimo is a core part of the broader Python
ecosystem and is a member of the NumFOCUS community, which includes projects
such as NumPy, SciPy, and Matplotlib.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" />