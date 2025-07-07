<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="Marimo Logo">
</p>

<p align="center">
  <em>A reactive Python notebook for reproducible, git-friendly, and deployable data science.</em>
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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI Version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda Version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"></a>
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads">
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo">
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

##  Marimo: The Reactive Python Notebook for Data Science

**Marimo is a revolutionary reactive Python notebook that transforms how you work with data, making your analysis reproducible, shareable, and interactive.** Designed for modern data workflows, Marimo enhances your productivity and collaboration.  [Learn more on GitHub](https://github.com/marimo-team/marimo).

**Key Features:**

*   🚀 **Batteries-Included:**  Replaces Jupyter, Streamlit, and more, offering a comprehensive data science environment.
*   ⚡️ **Reactive Execution:** Automatically updates dependent cells when you change code or interact with UI elements, ensuring data consistency.
*   🖐️ **Interactive UI:** Easily create interactive elements like sliders, tables, and plots that seamlessly integrate with your Python code.
*   🐍 **Git-Friendly:**  Notebooks are stored as standard `.py` files, making version control and collaboration a breeze.
*   🛢️ **Data-Focused:**  Built-in SQL support for querying dataframes, databases, and data warehouses.  Advanced features for dataframe filtering and searching.
*   🤖 **AI-Enhanced:** Integrate AI to generate code tailored for data work.
*   🔬 **Reproducible Results:** Guarantees consistent results with no hidden state, deterministic execution, and built-in package management.
*   🏃 **Executable Scripts:** Run your notebooks as standard Python scripts with command-line argument support.
*   🛜 **Shareable and Deployable:** Easily deploy notebooks as interactive web apps, slides, or run them in the browser via WASM.
*   🧩 **Code Reusability:** Import functions and classes between notebooks.
*   🧪 **Testable Notebooks:** Integrate testing frameworks like pytest to ensure code quality.
*   ⌨️ **Modern Editor:** Enhanced with GitHub Copilot, AI assistants, vim keybindings, a variable explorer, and other productivity features.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [quickstart](#quickstart) for a primer on our CLI._

## Dive into a Reactive Programming Environment

Marimo guarantees your notebook code, outputs, and program state stay consistent, tackling the issues common in traditional notebooks.

**What is Reactive Programming?**  Marimo automatically updates cells based on dependencies, minimizing the need for manual re-runs. When you delete a cell, Marimo removes its variables from memory.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Execution Example" />

<a name="expensive-notebooks"></a>
**Efficient Handling of Resource-Intensive Notebooks:** Configure Marimo to run cells on demand, marking them as stale to prevent unnecessary execution of expensive calculations.

**Synchronized UI Elements:** Interact with UI elements like sliders, dropdowns, dataframes, and chat interfaces, with dependent cells automatically updating with their latest values.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="Interactive UI Example" />

**Interactive DataFrames:** Explore, filter, sort, and page through large datasets directly within your notebooks.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Interactive DataFrames Example" />

**AI-Assisted Code Generation:** Use AI to generate code specialized for data work, including the ability to create entire notebooks from text descriptions.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI Code Generation Example" />

**SQL Integration:** Query data using SQL directly within your notebooks, with results accessible as Python dataframes.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Integration Example" />

**Dynamic Markdown:** Create interactive and dynamic narratives in markdown, powered by Python variables.

**Built-in Package Management:** Easily manage package dependencies, including installing packages on import, and serializing requirements within your notebooks.

**Deterministic Execution Order:** Code runs in a deterministic order based on variable references, allowing for more organized notebook structures.

**High-Performance Runtime:** Marimo optimizes execution by only running the necessary cells.

**All-in-One Solution:** Includes GitHub Copilot, AI assistants, Ruff code formatting, HTML export, fast code completion, a VS Code extension, an interactive dataframe viewer, and more.

## Quickstart

_The [marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq)
on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an overview of many features._

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

For advanced features, including SQL and AI completion:

```bash
pip install marimo[recommended]
```

**Getting Started:**

*   **Create/Edit Notebooks:** `marimo edit`
*   **Run as Web App:** `marimo run your_notebook.py`

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Marimo Run Example" />

*   **Execute as Script:** `python your_notebook.py`
*   **Convert Jupyter Notebooks:** `marimo convert your_notebook.ipynb > your_notebook.py` or use our [web interface](https://marimo.io/convert)

**Tutorials:**

```bash
marimo tutorial --help
```

## Need Help?

Consult the [FAQ](https://docs.marimo.io/faq.html) in our documentation.

## Explore Further

Marimo is easy to learn and offers advanced capabilities.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/embedding.gif" width="700px" alt="Marimo Embedding Example" />

Discover more through our [docs](https://docs.marimo.io), [examples](https://docs.marimo.io/examples/), and [gallery](https://marimo.io/gallery).

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive Concepts">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Inputs">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Plots">
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
        <img src="https://marimo.io/shield.svg" alt="Marimo Playground">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Marimo Playground">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Marimo Playground">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Marimo Playground">
      </a>
    </td>
  </tr>
</table>

## Contribute

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for guidance.

> Questions? [Join us on Discord](https://marimo.io/discord?ref=readme).

## Community

Join our community!

*   🌟 [Star on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Discord](https://marimo.io/discord?ref=readme)
*   📧 [Newsletter](https://marimo.io/newsletter)
*   ☁️ [Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [GitHub Discussions](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Twitter](https://twitter.com/marimo_io)
*   🎥 [YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS Affiliated Project:** Marimo is a part of the NumFOCUS community, alongside projects like NumPy, SciPy, and Matplotlib.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFocus Affiliated Project Logo" />

## Inspiration ✨

Marimo reimagines Python notebooks as reproducible, interactive programs, moving away from error-prone JSON formats.

We believe in better tools for better outcomes. Marimo is designed to be a superior environment for research, communication, and learning within the Python community.

Inspired by projects such as [Pluto.jl](https://github.com/fonsp/Pluto.jl), [ObservableHQ](https://observablehq.com/tutorials), and [Bret Victor's essays](http://worrydream.com/), Marimo contributes to the movement towards reactive dataflow programming. This includes ideas from [IPyflow](https://github.com/ipyflow/ipyflow), [streamlit](https://github.com/streamlit/streamlit), [TensorFlow](https://github.com/tensorflow/tensorflow), [PyTorch](https://github.com/pytorch/pytorch/tree/main), [JAX](https://github.com/google/jax), and [React](https://github.com/facebook/react).

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="Marimo Horizontal Logo">
</p>