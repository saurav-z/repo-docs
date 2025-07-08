<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="Marimo Logo">
</p>

<h1 align="center">Marimo: Reactive Python Notebooks for Data Science</h1>

<p align="center">
  <em>Transform your data analysis with Marimo, a revolutionary Python notebook that's reproducible, git-friendly, and easily deployable as scripts or interactive apps.</em>
</p>

<p align="center">
  <a href="https://docs.marimo.io" target="_blank"><strong>Docs</strong></a> ·
  <a href="https://marimo.io/discord?ref=readme" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://docs.marimo.io/examples/" target="_blank"><strong>Examples</strong></a> ·
  <a href="https://marimo.io/gallery/" target="_blank"><strong>Gallery</strong></a> ·
  <a href="https://www.youtube.com/@marimo-team/" target="_blank"><strong>YouTube</strong></a>
  · <a href="https://github.com/marimo-team/marimo" target="_blank"><strong>GitHub</strong></a>
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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"></a>
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>


## Key Features

*   🚀 **Batteries-Included**: Replaces Jupyter, Streamlit, and more, offering a comprehensive environment.
*   ⚡️ **Reactive**:  Changes in one cell automatically update dependent cells, ensuring consistency.
*   🖐️ **Interactive**: Integrate sliders, tables, and plots directly into your Python code without callbacks.
*   🐍 **Git-Friendly**:  Notebooks are stored as standard `.py` files for easy version control.
*   🛢️ **Data-Focused**:  Built-in SQL support for querying dataframes, databases, and warehouses.
*   🤖 **AI-Native**: Generate and enhance code using AI, specifically tailored for data work.
*   🔬 **Reproducible**: Guarantee deterministic execution, no hidden state, and built-in package management.
*   🏃 **Executable**:  Run notebooks as Python scripts, parameterized via the command line.
*   🛜 **Shareable**: Deploy notebooks as interactive web apps or slides, and run them in the browser via WASM.
*   🧩 **Reusable**: Import functions and classes between notebooks for streamlined development.
*   🧪 **Testable**:  Seamlessly integrate with pytest for thorough testing.
*   ⌨️ **Modern Editor**:  Enhance your workflow with GitHub Copilot, AI assistants, Vim keybindings, and more.

## Getting Started

Get up and running with Marimo in seconds!

```bash
pip install marimo && marimo tutorial intro
```

_Explore Marimo instantly with our [online playground](https://marimo.app/l/c7h6pz), which runs entirely in your browser!_

### Quickstart Commands

*   **Create/Edit Notebooks:** `marimo edit`
*   **Run as App:** `marimo run your_notebook.py`
*   **Execute as Script:** `python your_notebook.py`
*   **Convert Jupyter Notebooks:** `marimo convert your_notebook.ipynb > your_notebook.py`

## What is Marimo?

Marimo is a modern, reactive Python notebook designed for data scientists and engineers. It solves common problems associated with traditional notebooks by ensuring your code, outputs, and program state are always consistent. With Marimo, you can easily create reproducible analyses, interactive applications, and shareable presentations.  Explore the full power of Marimo on the [original GitHub repository](https://github.com/marimo-team/marimo).

### Core Principles

*   **Reactive Programming Environment**: Automatically updates cells when dependencies change.
*   **Support for Expensive Notebooks**: Configure the runtime to be lazy, marking cells as stale instead of running them automatically.
*   **Synchronized UI Elements**: Interact with UI elements, and see changes reflected in the notebook.
*   **Interactive DataFrames**: Page through, search, filter, and sort millions of rows.
*   **Generate Code with AI**: Create data-aware code with AI.
*   **Query Data with SQL**: Build SQL queries that depend on Python values.
*   **Dynamic Markdown**: Use markdown parameterized by Python variables.
*   **Built-in Package Management**: Install packages on import, and manage dependencies within notebooks.
*   **Deterministic Execution Order**: Notebooks are executed in a deterministic order, based on variable references.
*   **Performant Runtime**: Runs only those cells that need to be run.
*   **Batteries-Included**: Comes with quality-of-life features.

## Learn More

Discover the full potential of Marimo through our resources:

*   [Docs](https://docs.marimo.io)
*   [Usage Examples](https://docs.marimo.io/examples/)
*   [Gallery](https://marimo.io/gallery)

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive Notebook">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Interactive UI">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Plotting Example">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Output Examples">
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
        <img src="https://marimo.io/shield.svg" alt="Marimo playground">
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

## Contributing

Your contributions are highly valued!  See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for guidelines.

> Questions?  Reach out [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join our growing community:

*   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS Affiliated Project.** Marimo is part of the NumFOCUS community, which includes projects such as NumPy, SciPy, and Matplotlib.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFocus logo" />

## Inspiration

Marimo reimagines the Python notebook as a reproducible, interactive, and shareable program, moving away from error-prone JSON files.

Our goal is to provide the Python community with a better programming environment to research and communicate, experiment with code, and teach computational science.

We are inspired by projects like [Pluto.jl](https://github.com/fonsp/Pluto.jl), [ObservableHQ](https://observablehq.com/tutorials), and [Bret Victor's essays](http://worrydream.com/).  Marimo is part of a growing movement towards reactive dataflow programming, including projects like [IPyflow](https://github.com/ipyflow/ipyflow), [Streamlit](https://github.com/streamlit/streamlit), [TensorFlow](https://github.com/tensorflow/tensorflow), [PyTorch](https://github.com/pytorch/pytorch/tree/main), [JAX](https://github.com/google/jax), and [React](https://github.com/facebook/react).

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="Marimo Logo">
</p>