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

# marimo: The Reactive Python Notebook for Data Science and Beyond

**marimo is a revolutionary Python notebook that empowers you to create reproducible, interactive, and shareable data science projects, all while keeping your code clean and efficient.**

[<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="70px" align="right">](https://github.com/marimo-team/marimo)

**Key Features:**

*   🚀 **Reactive Notebooks:** Experience automatic updates!  Changes in one cell instantly update dependent cells.  No more manual re-running.
*   🐍 **Pure Python:** marimo notebooks are stored as `.py` files, making them git-friendly and easily version-controlled.
*   🖐️ **Interactive UI Components:**  Integrate sliders, dropdowns, tables, and more directly into your notebook with no callbacks required.
*   🛢️ **Data-Driven:** Effortlessly query and manipulate data with built-in SQL support, compatible with databases, dataframes, and more.
*   🤖 **AI-Powered:** Enhance your workflow with AI-driven code generation tailored for data work, plus notebook generation and other features.
*   🔬 **Reproducible and Reliable:**  Enjoy deterministic execution, no hidden state, and built-in package management for consistent results.
*   🏃 **Executable Scripts & Apps:**  Transform your notebooks into runnable Python scripts or deploy them as interactive web apps or slides.
*   🛜 **Shareable & Deployable:** Easily share your work, run notebooks in the browser, or deploy as interactive web applications.
*   🧩 **Reusable Code:** Import functions and classes from other notebooks to create modular and organized projects.
*   🧪 **Testable:** Integrate with `pytest` for robust testing of your notebooks.
*   ⌨️ **Modern Editor:** Benefit from features like GitHub Copilot integration, AI assistants, Vim keybindings, and a variable explorer for a streamlined coding experience.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [quickstart](#quickstart) for a primer on our CLI._

## What is marimo?

marimo is a reactive programming environment built on Python. Run a cell and marimo automatically runs the dependent cells to keep your notebook code, outputs, and program state consistent. 

**Key Benefits:**

*   **Eliminate Errors:** Avoid the pitfalls of manual cell re-running.
*   **No Hidden State:** Delete cells without leaving behind unwanted variables.
*   **Optimized Execution:** marimo only runs the cells that need to be updated.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Notebook Example" />

### Features for Efficiency

*   **Lazy Execution:** Prevent accidental execution of expensive cells by configuring the runtime to mark affected cells as stale instead of running them automatically.
*   **Synchronized UI Elements:** Interact with sliders, dropdowns, dataframe transformers, and chat interfaces; dependent cells are automatically re-run with the latest values.
*   **Interactive Dataframes:** Page, search, filter, and sort millions of rows with no coding required.
*   **AI-Powered Code Generation:** Generate code with an AI assistant that's specialized for working with data.
*   **SQL Integration:** Build SQL queries that depend on Python values and execute them against dataframes, databases, lakehouses, and more.
*   **Dynamic Markdown:** Use markdown parametrized by Python variables to create dynamic stories.
*   **Package Management:** Install packages on import and serialize package requirements.
*   **Deterministic Execution Order:** Notebooks are executed based on variable references, not cell positions.

## Quickstart

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with recommended dependencies, run:

```bash
pip install marimo[recommended]
```

**Commands:**

*   `marimo edit`: Create or edit notebooks
*   `marimo run your_notebook.py`: Run notebooks as web apps
*   `python your_notebook.py`: Execute notebooks as scripts
*   `marimo convert your_notebook.ipynb > your_notebook.py`: Convert Jupyter notebooks
*   `marimo tutorial --help`: List all available tutorials
*   Use [molab](https://molab.marimo.io/notebooks) for cloud-based notebook sharing.

## Learn More

marimo provides many features for a power-user experience.

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

Explore our [docs](https://docs.marimo.io/), [examples](https://docs.marimo.io/examples/), and [gallery](https://marimo.io/gallery) for more in-depth information.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) to get started.

> Have questions?  Join us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join the marimo community!

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

## Inspiration

marimo is a re-imagining of the Python notebook, offering a modern, reliable environment for Python programming.

Our work is inspired by projects like [Pluto.jl](https://github.com/fonsp/Pluto.jl), [ObservableHQ](https://observablehq.com/tutorials), and [Bret Victor's essays](http://worrydream.com/).

[**Back to Top**](#marimo-the-reactive-python-notebook-for-data-science-and-beyond)