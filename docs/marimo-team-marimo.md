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

## marimo: The Reactive Python Notebook for Modern Data Science

**marimo** is a revolutionary reactive Python notebook environment that transforms how you work with data; find out more about it on the [marimo GitHub repo](https://github.com/marimo-team/marimo). Experience reproducible results, seamless interactivity, and effortless deployment, all while using pure Python.

**Key Features:**

*   🚀 **Batteries-Included:** Replaces Jupyter, Streamlit, and more, streamlining your data science workflow.
*   ⚡️ **Reactive:** Automatically updates dependent cells when you modify code or interact with elements.
*   🖐️ **Interactive:** Bind sliders, tables, plots, and other UI elements to your Python code, without callbacks.
*   🐍 **Git-Friendly:** Stores notebooks as standard `.py` files for easy version control and collaboration.
*   🛢️ **Data-Centric:** Offers first-class support for SQL queries, dataframes, and databases.
*   🤖 **AI-Native:** Generate code with AI assistants tailored for data tasks.
*   🔬 **Reproducible:** Ensures consistent results with no hidden state and deterministic execution.
*   🏃 **Executable:** Run notebooks as Python scripts, parameterized by CLI arguments.
*   🛜 **Shareable:** Deploy your notebooks as interactive web apps or slides with ease, or run them in the browser via WASM.
*   🧩 **Reusable:** Import functions and classes between notebooks for modularity.
*   🧪 **Testable:** Integrates with `pytest` for robust testing.
*   ⌨️ **Modern Editor:** Enjoy features like GitHub Copilot integration, AI assistants, and Vim keybindings.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

## How marimo Works

marimo is a reactive programming environment that guarantees consistency between your notebook code, outputs, and program state. This eliminates the common issues associated with traditional notebooks like Jupyter.

**Benefits of Using marimo:**

*   **Reactive Updates:** Modify a cell, and marimo automatically runs the cells that depend on it, eliminating the need for manual re-runs.
*   **No Hidden State:** Delete a cell, and marimo removes its variables from memory, ensuring a clean and predictable state.
*   **Interactive UI:** Easily integrate UI elements such as sliders and dropdowns, and see the changes reflected in your code instantly.
*   **Optimized for Expensive Notebooks:** Configure marimo to mark cells as stale instead of re-running them, preventing the accidental execution of costly operations.
*   **Dynamic DataFrames:** Easily page through, search, filter, and sort millions of rows of data with a few clicks.
*   **Data-Aware AI Assistance:** Generate code with AI assistants to make data work easier.
*   **SQL Integration:** Build SQL queries that depend on Python variables and execute them against various data sources.
*   **Built-in Package Management:** Easily manage and install packages directly within your notebooks.
*   **Deterministic Execution:** Your notebooks run in a predictable order, making them easier to understand and debug.
*   **High Performance:** The marimo runtime runs only the necessary cells, optimizing performance.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" />

<a name="expensive-notebooks"></a>

## Get Started with marimo

*   **Installation:**
    *   Using `pip`: `pip install marimo`
    *   For recommended features like SQL support: `pip install marimo[recommended]`
    *   Install and run the introduction tutorial: `marimo tutorial intro`
*   **Create and Edit Notebooks:** Use the command `marimo edit` to start creating or editing.
*   **Run Apps:** Run your notebook as a web app by running: `marimo run your_notebook.py`
*   **Execute as Scripts:** Run your notebooks as standard Python scripts: `python your_notebook.py`
*   **Convert Jupyter Notebooks:** Convert your existing Jupyter notebooks to marimo with: `marimo convert your_notebook.ipynb > your_notebook.py`
*   **Tutorials:**
    *   List all tutorials: `marimo tutorial --help`

## Explore marimo

*   Visit the [FAQ](https://docs.marimo.io/faq.html) for answers to common questions.
*   Check out the [documentation](https://docs.marimo.io), [examples](https://docs.marimo.io/examples/), and [gallery](https://marimo.io/gallery) to learn more.

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

## Contribute

We welcome contributions! Check out [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Need Help?  Connect with us on [Discord](https://marimo.io/discord?ref=readme).

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

marimo is built upon a vision for a better way to create, share, and reproduce Python programs.

Inspired by projects like [Pluto.jl](https://github.com/fonsp/Pluto.jl) and [ObservableHQ](https://observablehq.com/tutorials), marimo is committed to improving the tools data scientists use.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px">
</p>