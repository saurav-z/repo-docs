<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="marimo Logo">
</p>

<p align="center">
  <em><strong>Revolutionize Your Python Workflow:</strong> Build interactive, reproducible, and shareable notebooks with marimo.</em>
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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI Version"/></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda Version"/></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"/></a>
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"/></a>
</p>

## What is marimo?

marimo is a cutting-edge, reactive Python notebook environment that transforms how you work with code and data. Unlike traditional notebooks, marimo guarantees consistency, reproducibility, and effortless sharing. It allows you to build interactive web apps, data visualizations, and data-driven narratives.

**[Visit the marimo repository on GitHub](https://github.com/marimo-team/marimo)**

## Key Features

*   🚀 **All-in-One Solution:**  Replaces multiple tools like Jupyter, Streamlit, and more, streamlining your workflow.
*   ⚡️ **Reactive Programming:**  Automatically updates dependent cells when you modify code or interact with UI elements, ensuring code and outputs are consistent.
*   🖐️ **Interactive UI Elements:**  Easily bind sliders, dropdowns, plots, and other elements to your Python code without the need for callbacks.
*   🐍 **Git-Friendly Notebooks:**  Store your notebooks as `.py` files for seamless version control and collaboration.
*   🛢️ **Data-Driven Design:** Built-in SQL support, enabling easy querying of dataframes, databases, warehouses, and lakehouses, plus dataframe filtering and searching.
*   🤖 **AI-Powered Code Generation:**  Generate Python code tailored for data work using integrated AI assistants.
*   🔬 **Reproducible Results:**  Guaranteed consistent program state with no hidden state, deterministic execution order, and built-in package management.
*   🏃 **Executable Scripts:** Execute notebooks directly as Python scripts, with support for command-line arguments.
*   🛜 **Shareable Applications:** Deploy your notebooks as interactive web apps, slides, or run them in the browser using WASM.
*   🧩 **Code Reusability:** Import functions and classes from one notebook to another, promoting modularity and efficiency.
*   🧪 **Testable Notebooks:**  Integrate pytest to test your notebooks and ensure code quality.
*   ⌨️ **Modern Editor:** Enjoy advanced features like GitHub Copilot integration, AI-powered code completion, Vim keybindings, a variable explorer, and more.

## Getting Started

```bash
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

### Quickstart

1.  **Installation:**

    ```bash
    pip install marimo  # or conda install -c conda-forge marimo
    marimo tutorial intro
    ```

    To install with recommended features, use:

    ```bash
    pip install marimo[recommended]
    ```
2.  **Create/Edit Notebooks:**

    ```bash
    marimo edit
    ```

3.  **Run as Web App:**

    ```bash
    marimo run your_notebook.py
    ```

4.  **Execute as Script:**

    ```bash
    python your_notebook.py
    ```

5.  **Convert Jupyter Notebooks:**

    ```bash
    marimo convert your_notebook.ipynb > your_notebook.py
    ```
    Or use our [web interface](https://marimo.io/convert).

## Learn More

Dive deeper into marimo's capabilities with our comprehensive resources:

*   [**Docs**](https://docs.marimo.io)
*   [**Examples**](https://docs.marimo.io/examples/)
*   [**Gallery**](https://marimo.io/gallery)

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive Notebook"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Interactive UI Elements"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Data Visualization"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Layout"/>
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
        <img src="https://marimo.io/shield.svg" alt="Live Playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Live Playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Live Playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Live Playground"/>
      </a>
    </td>
  </tr>
</table>
## Contribute

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

## Community

Join the marimo community and stay up-to-date:

*   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.** marimo is a part of the NumFOCUS community, which includes projects like NumPy, SciPy, and Matplotlib.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFocus affiliated project" />