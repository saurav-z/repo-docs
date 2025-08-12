<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="Marimo Logo">
</p>

<p align="center">
  <em>**Marimo: The reactive Python notebook that transforms how you code, share, and build data applications.**</em>
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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI Version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda Version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads">
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" >
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

## What is Marimo?

Marimo is a revolutionary reactive Python notebook designed for data scientists and developers.  It offers a modern, reproducible, and shareable environment for building interactive data applications.  **Explore the original repository: [marimo-team/marimo](https://github.com/marimo-team/marimo)**

## Key Features

*   🚀 **Batteries-Included:** Replaces Jupyter, Streamlit, and more, offering a streamlined development experience.
*   ⚡️ **Reactive Execution:** Automatically updates dependent cells when a cell's output changes, ensuring consistency.
*   🖐️ **Interactive UIs:** Easily create interactive elements like sliders, dropdowns, and charts directly in your notebooks without callbacks.
*   🐍 **Git-Friendly:** Stores notebooks as `.py` files, making version control simple and effective.
*   🛢️ **Data-Focused:** Native support for SQL queries, dataframe manipulation, and integration with databases, data warehouses, and data lakes.
*   🤖 **AI-Powered:** Generate cells with AI assistance, tailored for data work, improving productivity.
*   🔬 **Reproducible Results:** Guarantees reproducible results through deterministic execution and built-in package management.
*   🏃 **Executable Scripts:** Convert notebooks into Python scripts for easy execution and parameterization.
*   🛜 **Shareable Applications:** Deploy notebooks as interactive web apps or slides with built-in hosting options.
*   🧩 **Code Reusability:** Import functions and classes from one notebook to another for modularity.
*   🧪 **Testable Notebooks:** Integrate unit testing with pytest.
*   ⌨️ **Modern Editor:** Benefit from a modern code editor with features like GitHub Copilot integration, AI assistants, variable explorer, and more.

## Getting Started

Install marimo using pip or conda:

```bash
pip install marimo  # or conda install -c conda-forge marimo
```

or install with recommended dependencies

```bash
pip install marimo[recommended]
```

Then, launch the tutorial:

```bash
marimo tutorial intro
```

### Core Commands

*   `marimo edit`: Create and edit notebooks.
*   `marimo run your_notebook.py`: Run your notebook as a web app.
*   `python your_notebook.py`: Execute a notebook as a Python script.
*   `marimo convert your_notebook.ipynb > your_notebook.py`: Convert Jupyter notebooks to marimo notebooks.

## Learn More

*   **[marimo Documentation](https://docs.marimo.io)**
*   **[Examples](https://docs.marimo.io/examples/)**
*   **[Gallery](https://marimo.io/gallery)**
*   **[marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq) on YouTube**

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive Notebooks">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Interactive UI">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Data Visualization">
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
        <img src="https://marimo.io/shield.svg" alt="marimo cloud">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="marimo cloud">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="marimo cloud">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="marimo cloud">
      </a>
    </td>
  </tr>
</table>

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

## Community

*   🌟 [Star on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Join us on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

## Affiliations

A **NumFOCUS** affiliated project, part of the Python ecosystem.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS Affiliated Project" />

## Inspiration
Marimo is inspired by Pluto.jl, ObservableHQ, and Bret Victor's essays, reflecting a movement toward reactive dataflow programming.