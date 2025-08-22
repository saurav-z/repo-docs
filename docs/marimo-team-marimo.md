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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

# marimo: The Reactive Python Notebook for Data Science

**marimo** is a revolutionary reactive Python notebook designed for data scientists, offering a reproducible, git-friendly, and deployable environment for creating interactive analyses, applications, and presentations. [Learn more on GitHub](https://github.com/marimo-team/marimo).

## Key Features

*   🚀 **Batteries-included:** Combines the functionalities of `Jupyter`, `Streamlit`, `Jupytext`, `ipywidgets`, `papermill`, and more.
*   ⚡️ **Reactive Execution:** Automatically runs dependent cells when a cell's input changes, or marks them as stale, ensuring consistent code and output.
*   🖐️ **Interactive Elements:** Easily integrate sliders, tables, plots, and other UI elements directly into your Python code without the need for callbacks.
*   🐍 **Git-Friendly:** Stores notebooks as `.py` files for seamless version control and collaboration.
*   🛢️ **Data-Focused:** Seamlessly query data from dataframes, databases, warehouses, and lakehouses using SQL, as well as filter and search dataframes.
*   🤖 **AI-Powered:** Generate code cells with AI tailored for data work, enhancing your workflow.
*   🔬 **Reproducible & Reliable:** Guarantees no hidden state, deterministic execution, and built-in package management for reliable results.
*   🏃 **Executable as Scripts:** Run notebooks as standard Python scripts, optionally parameterized via the command line.
*   🛜 **Shareable & Deployable:** Deploy your notebooks as interactive web apps or presentations, and even run them in the browser using WASM.
*   🧩 **Reusable Code:** Import functions and classes from other notebooks to promote code reusability.
*   🧪 **Testable:** Supports running tests with `pytest` directly on your notebooks.
*   ⌨️ **Modern Editor:** Includes a modern editor with features like GitHub Copilot integration, AI assistants, Vim keybindings, a variable explorer, and more for an enhanced coding experience.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

## Core Concepts

marimo guarantees your notebook code, outputs, and program state are consistent, solving common issues found in traditional notebooks.

### Reactive Programming Environment

Run a cell, and marimo automatically re-runs dependent cells, removing the need to manually rerun cells. Delete a cell and marimo scrubs its variables from program memory, eliminating hidden state.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive execution example" />

### Handling Expensive Notebooks

Configure the runtime to be lazy, marking affected cells as stale to prevent accidental execution of expensive cells, while still guaranteeing program state consistency.

### Synchronized UI Elements

Interactive UI elements like sliders, dropdowns, dataframe transformers, and chat interfaces automatically trigger the re-running of cells using their latest values.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI elements example" />

### Interactive Dataframes

Quickly page through, search, filter, and sort millions of rows, without writing code.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Interactive dataframe example" />

### AI-Assisted Code Generation

Generate code with an AI assistant specialized for data work, with context about your variables in memory; [zero-shot entire notebooks](https://docs.marimo.io/guides/generate_with_ai/text_to_notebook/). Customize the system prompt, bring your own API keys, or use local models.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI code generation example" />

### SQL Queries in Notebooks

Build SQL queries that depend on Python variables and execute them against dataframes, databases, lakehouses, CSVs, and more.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL query example" />

### Additional Features

*   **Dynamic Markdown:** Use markdown parametrized by Python variables to tell dynamic stories.
*   **Built-in Package Management:** Install packages on import and serialize package requirements in notebook files.
*   **Deterministic Execution Order:** Code executes in a deterministic order, based on variable references.
*   **Performant Runtime:** Executes only the cells that need to be run.
*   **Batteries-Included:** Comes with GitHub Copilot, AI assistants, Ruff code formatting, HTML export, fast code completion, an interactive dataframe viewer, and more.

## Quickstart

_The [marimo concepts
playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq)
on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an
overview of many features._

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with additional dependencies that unlock SQL cells, AI completion, and more:

```bash
pip install marimo[recommended]
```

**Commands:**

*   **Create/Edit Notebooks:** `marimo edit`
*   **Run as Web App:** `marimo run your_notebook.py`

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Run app example" />

*   **Execute as Script:** `python your_notebook.py`
*   **Convert Jupyter Notebooks:** `marimo convert your_notebook.ipynb > your_notebook.py`  or use the [web interface](https://marimo.io/convert).
*   **Tutorials:** `marimo tutorial --help`
*   **Cloud-Based Notebooks:** Use [molab](https://molab.marimo.io/notebooks).

## Getting Help

Refer to the [FAQ](https://docs.marimo.io/faq.html) at our docs for answers.

## Learn More

Explore the power of marimo with our [docs](https://docs.marimo.io), [usage examples](https://docs.marimo.io/examples/), and [gallery](https://marimo.io/gallery).

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Tutorial">
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
        <img src="https://marimo.io/shield.svg" alt="marimo online playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="marimo example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="marimo example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="marimo example"/>
      </a>
    </td>
  </tr>
</table>

## Contributing

Contributions are welcome! Review [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Have questions? Join us [on Discord](https://marimo.io/discord?ref=readme).

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

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS affiliated project" />

## Inspiration ✨

marimo reimagines the Python notebook as a reproducible, interactive, and shareable Python program, offering a superior alternative to traditional, error-prone JSON-based notebooks.

We aim to provide the Python community with a better programming environment for research, communication, and learning.

Inspired by [Pluto.jl](https://github.com/fonsp/Pluto.jl),
[ObservableHQ](https://observablehq.com/tutorials), and
[Bret Victor's essays](http://worrydream.com/), marimo is part of the trend toward reactive dataflow programming, similar to [IPyflow](https://github.com/ipyflow/ipyflow), [streamlit](https://github.com/streamlit/streamlit),
[TensorFlow](https://github.com/tensorflow/tensorflow),
[PyTorch](https://github.com/pytorch/pytorch/tree/main),
[JAX](https://github.com/google/jax), and
[React](https://github.com/facebook/react).

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo logo">
</p>
```

Key improvements:

*   **SEO Optimization:** Added relevant keywords like "Python notebook," "reactive," "data science," "interactive," "reproducible," "Git-friendly," and "deployable."
*   **Clear Structure:**  Uses headings to organize the information logically.
*   **Concise Summary:** Starts with a compelling one-sentence hook.
*   **Emphasis on Benefits:** Highlights key features and benefits, rather than just describing features.
*   **Use of Alt Text:**  Added alt text to all image tags for accessibility and SEO.
*   **Clean Formatting:**  Improved readability with better spacing and use of bullet points.
*   **Comprehensive Overview:**  Covers installation, basic usage, and resources for learning more.
*   **Call to Action:** Encourages users to engage with the project (star, chat, subscribe, etc.)

This improved README is much more likely to rank well in search results and effectively communicate the value of marimo to potential users.