<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg">
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

# marimo: The Reactive Python Notebook for Data Scientists and Developers

marimo is a cutting-edge, reactive Python notebook that transforms how you work with data, making your projects reproducible, shareable, and deployable as scripts or interactive apps.  [Explore marimo on GitHub](https://github.com/marimo-team/marimo).

## Key Features

*   **Reactive & Intuitive:** Run a cell, and marimo automatically updates dependent cells, ensuring your code and outputs stay consistent.
*   **Interactive:** Bind sliders, tables, plots, and more to Python without complex callbacks, making your notebooks truly interactive.
*   **Git-Friendly:** marimo notebooks are stored as `.py` files, seamlessly integrating with version control systems.
*   **Data-Focused:** Query dataframes, databases, and warehouses with SQL, and easily filter and search your data.
*   **AI-Powered:** Generate Python code directly within your notebook using AI assistants tailored for data tasks.
*   **Reproducible & Testable:** No hidden state, deterministic execution, and built-in package management ensures reliability. Test your notebooks with pytest!
*   **Executable & Shareable:** Execute notebooks as scripts or deploy them as interactive web apps or slides, or run in the browser via WASM.
*   **Modern Editor:** Benefit from a modern editor with GitHub Copilot, AI assistants, Vim keybindings, and a variable explorer.
*   **Batteries Included:** A complete solution that replaces Jupyter, Streamlit, and more.
*   **Reusable Notebooks:** Import functions and classes from one notebook to another.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [quickstart](#quickstart) for a primer on our CLI._

## Core Concepts

marimo is a reactive programming environment that guarantees consistency between your code, outputs, and program state.

*   **Reactive Programming:** Automatic updates of dependent cells when a cell is run.
*   **UI Element Synchronization:** Seamless interaction with UI elements like sliders, dropdowns, and chat interfaces, with instant updates in your notebook.
*   **Interactive DataFrames:**  Effortlessly page, search, filter, and sort millions of rows with zero code!
*   **AI-Enhanced Development:**  Generate data-aware code with AI assistants.
*   **SQL Integration:**  Query data using SQL directly within your Python notebooks, even against databases.

## Quickstart

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

For advanced features, install with:

```bash
pip install marimo[recommended]
```

**Key Commands:**

*   `marimo edit`: Create and edit notebooks.
*   `marimo run your_notebook.py`: Run your notebook as an interactive web app.
*   `python your_notebook.py`: Execute a notebook as a standard Python script.
*   `marimo convert your_notebook.ipynb > your_notebook.py`: Convert Jupyter notebooks to marimo notebooks.

## Learn More

Dive deeper into marimo:

*   [marimo Documentation](https://docs.marimo.io)
*   [marimo Examples](https://docs.marimo.io/examples/)
*   [marimo Gallery](https://marimo.io/gallery)

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

We welcome contributions! Check out [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

>   Need help?  Join us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Connect with the marimo community:

*   [Star us on GitHub](https://github.com/marimo-team/marimo)
*   [Join us on Discord](https://marimo.io/discord?ref=readme)
*   [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   [Join our Cloud Waitlist](https://marimo.io/cloud)
*   [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   [Follow us on Twitter](https://twitter.com/marimo_io)
*   [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS Affiliated Project.** marimo is a core part of the broader Python ecosystem and is a member of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" />

## Inspiration

marimo is inspired by projects like Pluto.jl and ObservableHQ, and the ideas of reactive and dataflow programming. We aim to provide a better programming environment for research, communication, and experimentation.