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

## Marimo: The Reactive Python Notebook for Data Science 

**Marimo** is revolutionizing the way you work with Python by offering a reactive notebook experience that is reproducible, git-friendly, and deployable as scripts or apps.  [Explore the code on GitHub](https://github.com/marimo-team/marimo).

**Key Features:**

*   🚀 **Batteries-included:** Replaces tools like Jupyter, Streamlit, and more.
*   ⚡️ **Reactive:** Automatically updates dependent cells when you change code or interact with UI elements.
*   🖐️ **Interactive:** Easily bind UI elements like sliders and dataframes to your Python code, without complex callbacks.
*   🐍 **Git-friendly:**  Notebooks are stored as `.py` files for easy version control.
*   🛢️ **Designed for Data:** Offers first-class support for SQL, dataframe manipulation, and data visualization.
*   🤖 **AI-native:**  Leverage AI to generate code within your notebooks.
*   🔬 **Reproducible:** Ensures consistent results with deterministic execution and built-in package management.
*   🏃 **Executable:** Run notebooks as standard Python scripts with CLI argument support.
*   🛜 **Shareable:** Deploy notebooks as interactive web apps or presentations.
*   🧩 **Reusable:** Import functions and classes between notebooks.
*   🧪 **Testable:** Integrate with pytest for robust testing.
*   ⌨️ **Modern Editor:** Includes features like GitHub Copilot, AI assistants, and Vim keybindings for an enhanced coding experience.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

## Core Concepts

Marimo provides a reactive programming environment that ensures code, outputs, and state consistency. Unlike traditional notebooks, marimo automatically updates dependent cells when a cell is modified, eliminating the need for manual re-running and reducing errors.

**Key Features:**

*   **Reactive Programming:**  Automatically re-executes dependent cells, maintaining consistency.
*   **Configurable Runtime:** Supports lazy execution for computationally expensive notebooks.
*   **Synchronized UI Elements:**  Interactive elements like sliders and dropdowns automatically update the connected code.
*   **Interactive DataFrames:** Provides in-notebook capabilities for data exploration, including paging, searching, filtering, and sorting.
*   **AI-Powered Code Generation:** Integrated AI assistants can generate data-aware code and even entire notebooks.
*   **SQL Integration:** Built-in SQL engine enables querying of dataframes, databases, and other data sources.
*   **Dynamic Markdown:**  Create dynamic markdown based on your Python variables.
*   **Package Management:**  Built-in support for package management, including installing packages on import and serializing dependencies within notebook files.
*   **Deterministic Execution:** Notebooks run in a deterministic order based on variable references.
*   **Optimized Performance:** The runtime executes only the necessary cells.
*   **Comprehensive Features:**  Includes features like GitHub Copilot, VS Code extension, interactive dataframe viewer, and much more.

## Quickstart

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with recommended dependencies:

```bash
pip install marimo[recommended]
```

**Commands:**

*   **Create/Edit:** `marimo edit`
*   **Run as App:** `marimo run your_notebook.py`
*   **Execute as Script:** `python your_notebook.py`
*   **Convert Jupyter Notebook:** `marimo convert your_notebook.ipynb > your_notebook.py` or use the [web interface](https://marimo.io/convert).
*   **Tutorials:** `marimo tutorial --help`

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" />

## Learn More

Explore the [docs](https://docs.marimo.io), [examples](https://docs.marimo.io/examples/), and [gallery](https://marimo.io/gallery) to discover more features and capabilities.

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

Contributions are welcome!  See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Questions?  Join us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join the Marimo community!

*   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.** Marimo is a part of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" />

## Inspiration ✨

Marimo aims to reinvent the Python notebook as a more robust and reliable tool.

We believe better tools lead to better outcomes, so with marimo, we hope to provide a better environment for research, communication, and learning in the Python community.