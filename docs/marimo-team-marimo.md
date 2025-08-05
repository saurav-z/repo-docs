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
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Traditional_Chinese.md" target="_blank"><b>繁體中文</b></a>
  <b> | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Chinese.md" target="_blank"><b>简体中文</b></a>
  <b> | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Japanese.md" target="_blank"><b>日本語</b></a>
  <b> | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Spanish.md" target="_blank"><b>Español</b></a>
</p>

<p align="center">
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI version"/></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda version"/></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"/></a>
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"/></a>
</p>

## marimo: Revolutionizing Python Notebooks with Reactivity, Reproducibility, and More

**marimo** is a next-generation Python notebook designed to be reactive, reproducible, and easily deployable.  It's a powerful alternative to traditional notebooks like Jupyter, offering a more robust and streamlined development experience.  [Check out the original repository for more details.](https://github.com/marimo-team/marimo)

**Key Features:**

*   🚀 **Batteries-Included:** Replaces many tools like `jupyter`, `streamlit`, `jupytext`, `ipywidgets`, and more.
*   ⚡️ **Reactive Execution:**  Changes in one cell automatically update dependent cells, ensuring consistency.
*   🖐️ **Interactive UI Elements:**  Easily integrate sliders, tables, plots, and other UI elements with Python code, without callbacks.
*   🐍 **Git-Friendly:**  Notebooks are stored as standard `.py` files for seamless version control.
*   🛢️ **Designed for Data:**  Query dataframes, databases, and warehouses directly using SQL, along with advanced dataframe filtering and search capabilities.
*   🤖 **AI-Powered:** Generate cells with AI tailored for data work.
*   🔬 **Reproducible Results:** Eliminates hidden state, and offers deterministic execution and built-in package management.
*   🏃 **Executable Scripts:** Execute notebooks as standard Python scripts, parameterized by CLI arguments.
*   🛜 **Shareable Apps:** Deploy notebooks as interactive web apps or slides, and even run them in the browser via WASM.
*   🧩 **Code Reusability:** Import functions and classes from one notebook to another.
*   🧪 **Testability:** Easily run pytest on your notebooks.
*   ⌨️ **Modern Editor:** Includes features like GitHub Copilot integration, AI assistants, Vim keybindings, and a variable explorer, providing a modern editing experience.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

## Key Benefits & Functionality

marimo goes beyond the limitations of traditional notebooks by:

*   **Guaranteed Consistency:** Ensuring that code, outputs, and program state are always synchronized.
*   **Reactive Programming Environment:** Automatically re-running dependent cells when a cell's variables change.
*   **Support for Expensive Notebooks:** Allowing you to configure the runtime to be lazy, marking affected cells as stale instead of automatically running them, preventing unintended execution of costly operations.
*   **Synchronized UI elements:** Interact with UI elements like sliders, dropdowns, dataframe transformers, and chat interfaces.
*   **Interactive Dataframes:**  Quickly page through, search, filter, and sort millions of rows directly within the notebook.
*   **AI-Assisted Code Generation:** Leveraging AI assistants specialized for data tasks to generate code.
*   **Integrated SQL Support:**  Seamlessly query data from various sources using SQL, with results returned as Python dataframes.
*   **Dynamic Markdown:** Use markdown parametrized by Python variables to tell dynamic stories that depend on Python data.
*   **Built-in Package Management:** Easily install and manage packages within your notebooks.
*   **Deterministic Execution Order:** Execute notebooks in a deterministic order based on variable dependencies.
*   **Performant Runtime:** Efficiently runs only the cells that require execution based on code analysis.

## Quickstart

*The [marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq) on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an overview of many features.*

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```
To install with additional dependencies that unlock SQL cells, AI completion, and more, run
```bash
pip install marimo[recommended]
```

**Create/Edit Notebooks:**

```bash
marimo edit
```

**Run as Web App:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="running marimo app"/>

**Execute as Script:**

```bash
python your_notebook.py
```

**Convert Jupyter Notebooks:**

```bash
marimo convert your_notebook.ipynb > your_notebook.py
```
Or use our [web interface](https://marimo.io/convert).

**Tutorials:**
```bash
marimo tutorial --help
```

**Share Cloud-Based Notebooks:** Use [molab](https://molab.marimo.io/notebooks) to create and share links.

## Learn More

marimo offers a great starting point and a wide range of functionality for power users.

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="reactive"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="UI elements"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="plotting"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="layouts"/>
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
        <img src="https://marimo.io/shield.svg" alt="playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="playground"/>
      </a>
    </td>
  </tr>
</table>

Explore the [documentation](https://docs.marimo.io), [usage examples](https://docs.marimo.io/examples/), and the [gallery](https://marimo.io/gallery) to unlock more features.

## Contribute

Contributions are welcome!  Refer to [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Questions? Reach out to us [on Discord](https://marimo.io/discord?ref=readme).

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

**NumFOCUS Affiliated Project:** marimo is part of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS"/>

## Inspiration

marimo reimagines the Python notebook as a reproducible, interactive, and shareable Python program, promoting a more efficient and reliable coding experience.  Inspired by projects like Pluto.jl and ObservableHQ, marimo embraces reactive dataflow programming.