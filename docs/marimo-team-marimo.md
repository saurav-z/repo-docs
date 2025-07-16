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

**marimo** is a modern Python notebook that transforms how you interact with code, making your data science workflows more efficient, reproducible, and shareable.  [Explore marimo on GitHub](https://github.com/marimo-team/marimo).

## Key Features

*   🚀 **Batteries-Included:** Replaces Jupyter, Streamlit, and more, providing a comprehensive environment.
*   ⚡️ **Reactive:** Automatically updates dependent cells when you change code or UI elements, ensuring consistency.
*   🖐️ **Interactive:**  Easily bind sliders, tables, plots, and other UI elements to your Python code without complex callbacks.
*   🐍 **Git-Friendly:**  Notebooks are stored as standard `.py` files, making version control simple.
*   🛢️ **Designed for Data:** Query dataframes, databases, and warehouses with built-in SQL support and easily filter and search dataframes.
*   🤖 **AI-Native:** Generate cells with AI tailored for data work.
*   🔬 **Reproducible:** No hidden state, deterministic execution, and built-in package management for reliable results.
*   🏃 **Executable:** Run your notebooks as standalone Python scripts, customizable with CLI arguments.
*   🛜 **Shareable:** Deploy notebooks as interactive web apps, slideshows, or even run them in the browser using WASM.
*   🧩 **Reusable:** Import functions and classes between notebooks for cleaner, more organized code.
*   🧪 **Testable:**  Integrate with pytest for robust testing of your notebooks.
*   ⌨️ **Modern Editor:** Enjoy a modern editor with features like GitHub Copilot, AI assistants, Vim keybindings, and a variable explorer.

## Getting Started

Install marimo using pip:

```bash
pip install marimo && marimo tutorial intro
```

For additional features like SQL and AI completion, install with:

```bash
pip install marimo[recommended]
```

### Core Commands:

*   `marimo edit`: Create and edit notebooks.
*   `marimo run your_notebook.py`: Run your notebook as a web app.
*   `python your_notebook.py`: Execute a notebook as a script.
*   `marimo convert your_notebook.ipynb > your_notebook.py`: Convert Jupyter notebooks to marimo format.

Explore the [Quickstart](#quickstart) for a CLI primer.

## A Deeper Dive into marimo

marimo offers a cutting-edge reactive programming environment designed to simplify and improve data science workflows.

**Reactive Programming Environment:** marimo automatically runs dependent cells when a change is made, reducing errors and ensuring consistency.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" />

**Compatible with Expensive Notebooks:** Configure marimo to mark cells as stale instead of running them to prevent accidental execution of expensive calculations.

**Synchronized UI Elements:** Interact with sliders, dropdowns, dataframe transformers, and chat interfaces, and see the cells that use them update instantly.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" />

**Interactive DataFrames:** Page through, search, filter, and sort millions of rows with lightning speed, right within your notebook.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" />

**AI-Powered Code Generation:** Use an AI assistant to generate code tailored for data tasks, with context about your variables.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" />

**SQL Integration:** Build SQL queries that depend on Python values and execute them against dataframes, databases, lakehouses, CSVs, Google Sheets, or anything else.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" />

## Quickstart

*   **Installation:** (See above) `pip install marimo` or `conda install -c conda-forge marimo`
*   **Create/Edit:** `marimo edit`
*   **Run as App:** `marimo run your_notebook.py`
*   **Execute as Script:** `python your_notebook.py`
*   **Convert from Jupyter:** `marimo convert your_notebook.ipynb > your_notebook.py`

## Explore Further

Enhance your knowledge with these resources:

*   [marimo Concepts Playlist on YouTube](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq)
*   [marimo Docs](https://docs.marimo.io)
*   [Example Usage](https://docs.marimo.io/examples/)
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

Contributions are welcomed and valued!  Check out [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for guidelines.

## Connect with the Community

*   ⭐ [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Join our Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS Affiliated Project.** marimo is part of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" />

## Inspiration

marimo is inspired by projects like Pluto.jl and ObservableHQ, and is part of a movement towards reactive dataflow programming.