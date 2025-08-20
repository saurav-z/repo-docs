<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="Marimo Logo">
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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda Version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"></a>
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads">
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" >
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

# Marimo: The Reactive Python Notebook for Data Science

**Marimo is a next-generation Python notebook designed for data science, offering a reactive, reproducible, and shareable environment for your projects.**  ([View on GitHub](https://github.com/marimo-team/marimo))

## Key Features

*   🚀 **Batteries-Included:** Replaces multiple tools like Jupyter, Streamlit, and more, streamlining your workflow.
*   ⚡️ **Reactive:** Automatically updates dependent cells when you run a cell or interact with a UI element.
*   🖐️ **Interactive:** Easily create interactive elements like sliders, dropdowns, and plots without complex callbacks.
*   🐍 **Git-Friendly:** Stores notebooks as pure Python files, making version control simple.
*   🛢️ **Data-Focused:** Built-in SQL support for querying dataframes, databases, and data warehouses.
*   🤖 **AI-Native:** Generate code and entire notebooks with AI, tailored for data tasks.
*   🔬 **Reproducible:** Guarantees consistent code and outputs through deterministic execution and package management.
*   🏃 **Executable:** Run notebooks as standalone Python scripts with command-line argument support.
*   🛜 **Shareable:** Deploy notebooks as interactive web apps, slides, or run them in the browser via WASM.
*   🧩 **Reusable:** Import functions and classes between notebooks for efficient code reuse.
*   🧪 **Testable:** Easily integrate testing with pytest.
*   ⌨️ **Modern Editor:** Benefit from a modern editor with GitHub Copilot, AI assistants, and more.

## Getting Started

```bash
pip install marimo
marimo tutorial intro
```

For an enhanced experience, install with recommended dependencies:

```bash
pip install marimo[recommended]
```

### Core Commands

*   **Create/Edit:** `marimo edit`
*   **Run as App:** `marimo run your_notebook.py`
*   **Execute as Script:** `python your_notebook.py`
*   **Convert Jupyter Notebooks:** `marimo convert your_notebook.ipynb > your_notebook.py`

### Explore Further

*   **Tutorials:** `marimo tutorial --help`
*   **Cloud Notebooks:** Use [molab](https://molab.marimo.io/notebooks) to create and share cloud-based notebooks.

## A Deeper Dive

Marimo provides a reactive programming environment that ensures code, outputs, and program state remain consistent, solving many issues associated with traditional notebooks.

*   **Reactive Execution:** Changes in one cell automatically trigger updates in dependent cells.
*   **UI Element Synchronization:** UI elements like sliders and dropdowns seamlessly integrate with your code.
*   **Interactive DataFrames:** Easily explore, search, filter, and sort large datasets directly within your notebook.
*   **AI-Powered Code Generation:** Leverage AI to generate code tailored for data analysis tasks.
*   **SQL Integration:** Query data using SQL directly within your Python notebooks.
*   **Dynamic Markdown:** Create dynamic stories with markdown that updates based on your Python data.
*   **Built-in Package Management:** Simplifies package installation and dependency management.
*   **Deterministic Execution:** Notebooks execute in a consistent order based on variable dependencies.
*   **Performant Runtime:** Only the necessary cells are executed, improving efficiency.

## Learn More

Explore the [documentation](https://docs.marimo.io), [examples](https://docs.marimo.io/examples/), and [gallery](https://marimo.io/gallery) to discover the full potential of Marimo.

## Contributing

We welcome contributions! Review [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) to get started.

## Community

*   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.**

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS Affiliated Project">