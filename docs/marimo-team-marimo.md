<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="Marimo Logo">
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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI Version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda Version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord" /></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"/></a>
</p>

# Marimo: The Reactive Python Notebook You'll Love

**Marimo is a revolutionary Python notebook that transforms how you work with code, data, and apps, offering a reactive, reproducible, and shareable experience.** ([See the original repo](https://github.com/marimo-team/marimo))

## Key Features

*   🚀 **Batteries-Included**: Replaces Jupyter, Streamlit, and more, providing a comprehensive environment.
*   ⚡️ **Reactive Programming**: Automatic updates – run a cell, and dependent cells update automatically.
*   🖐️ **Interactive**: Bind sliders, tables, and plots directly to Python code with no callbacks required.
*   🐍 **Git-Friendly**: Notebooks are stored as `.py` files, making them version control-friendly.
*   🛢️ **Data-Focused**: Includes first-class SQL support and tools for dataframe manipulation.
*   🤖 **AI-Native**: Generate cells with AI tailored for data work.
*   🔬 **Reproducible**: No hidden state, ensuring deterministic execution and built-in package management.
*   🏃 **Executable**: Run notebooks as Python scripts with CLI argument support.
*   🛜 **Shareable**: Deploy interactive web apps, slides, and run in the browser via WASM.
*   🧩 **Reusable**: Import functions and classes between notebooks.
*   🧪 **Testable**: Integrate with pytest for notebook testing.
*   ⌨️ **Modern Editor**: Get GitHub Copilot integration, AI assistants, Vim keybindings, and more.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

## A Powerful Reactive Environment

marimo ensures your notebook code, outputs, and state are always synchronized.

**Reactive Programming Environment:** Marimo automatically runs dependent cells when a cell changes, and deletes variables when a cell is deleted.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Example">

**Compatible with Expensive Notebooks**: Configure runtime to be lazy and mark cells as stale.

**Synchronized UI Elements:** Interact with UI elements (sliders, dropdowns, etc.) and see the changes reflected in your code instantly.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI Example">

**Interactive Dataframes**: Page, search, filter, and sort millions of rows without writing any code.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Dataframe Example">

**AI-Assisted Code Generation**: Generate code using AI that's specialized for data, with context about your variables in memory; zero-shot entire notebooks.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI Code Generation Example">

**SQL Integration**: Build and execute SQL queries against dataframes, databases, or cloud warehouses.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Example">

**Dynamic Markdown**: Create dynamic, data-driven markdown content.

**Built-in Package Management**: Install packages directly within your notebooks with support for all major package managers.

**Deterministic Execution Order**: Execute notebooks in a logical order based on variable references.

**Performant Runtime**: Runs only the cells needed by statically analyzing your code.

**Batteries-Included**: Comes with GitHub Copilot, AI assistants, Ruff code formatting, HTML export, and a VS Code extension.

## Quickstart

_The [marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq)
on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an
overview of many features._

**Installation**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with additional dependencies, run:

```bash
pip install marimo[recommended]
```

**Create Notebooks:**

```bash
marimo edit
```

**Run Apps:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="App Example" />

**Execute as Scripts:**

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

**Share Cloud-Based Notebooks:** Use [molab](https://molab.marimo.io/notebooks).

## Questions?

See the [FAQ](https://docs.marimo.io/faq.html) in the docs.

## Learn More

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive Tutorial" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Inputs Tutorial" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Plotting Tutorial" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Layout Tutorial" />
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
        <img src="https://marimo.io/shield.svg" alt="Marimo Playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Marimo Playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Marimo Playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Marimo Playground"/>
      </a>
    </td>
  </tr>
</table>

## Contributing

See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Questions? Reach out [on Discord](https://marimo.io/discord?ref=readme).

## Community

*   🌟 [Star on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.**

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS Affiliated Project"/>

## Inspiration ✨

Marimo is a re-imagining of the Python notebook as a reproducible, interactive, and shareable Python program.  We aim to provide the Python community with a better environment for research, experimentation, and communication.  Our inspiration comes from many places, including Pluto.jl, ObservableHQ, and Bret Victor's essays.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="Marimo Logo">
</p>