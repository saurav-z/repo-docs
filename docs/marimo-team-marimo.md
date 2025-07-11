<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="marimo logo">
</p>

<p align="center">
  <em>A reactive Python notebook that's reproducible, git-friendly, and deployable as scripts or apps.</em>
</p>

<p align="center">
  <a href="https://docs.marimo.io" target="_blank"><strong>Docs</strong></a> |
  <a href="https://marimo.io/discord?ref=readme" target="_blank"><strong>Discord</strong></a> |
  <a href="https://docs.marimo.io/examples/" target="_blank"><strong>Examples</strong></a> |
  <a href="https://marimo.io/gallery/" target="_blank"><strong>Gallery</strong></a> |
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
  <a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI version"></a>
  <a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda version"></a>
  <a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"></a>
  <img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads">
  <img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" >
  <a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

## marimo: The Reactive Python Notebook for Data Science and Beyond

**marimo** is a revolutionary, reactive Python notebook that transforms how you work with code, data, and interactive applications. It's reproducible, git-friendly, and easily deployable as both scripts and web apps.  [Explore the marimo project on GitHub](https://github.com/marimo-team/marimo).

**Key Features:**

*   🚀 **Batteries-Included:** Replaces Jupyter, Streamlit, and more, providing a comprehensive data science environment.
*   ⚡️ **Reactive Execution:** Run a cell, and marimo automatically and efficiently updates dependent cells.
*   🖐️ **Interactive UIs:** Bind UI elements (sliders, dropdowns, charts, etc.) directly to Python variables with no callbacks needed.
*   🐍 **Git-Friendly:** Store your notebooks as `.py` files, making version control simple.
*   🛢️ **Data-Centric:** Seamlessly work with data using SQL queries, interactive dataframes, and more.
*   🤖 **AI-Enhanced:** Leverage AI to generate code and assist with your data analysis workflow.
*   🔬 **Reproducible:** Achieve consistent results with no hidden state, deterministic execution, and built-in package management.
*   🏃 **Executable as Scripts:** Execute your notebooks directly as Python scripts.
*   🛜 **Shareable & Deployable:**  Easily share your work as interactive web apps, slide decks, or via WASM.
*   🧩 **Code Reusability:**  Import functions and classes between notebooks for modularity.
*   🧪 **Testable:**  Integrate pytest for robust testing of your notebooks.
*   ⌨️ **Modern Editor:** Benefit from features like GitHub Copilot, AI assistants, and more.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [quickstart](#quickstart) for a primer on our CLI._

## Core Concepts

marimo's reactive engine ensures your code, outputs, and program state remain synchronized.

### Reactive Programming

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Programming GIF" />

marimo's core is a reactive engine. Change a cell, and marimo efficiently updates dependent cells, eliminating manual re-runs and hidden state.

### Handling Expensive Notebooks

marimo lets you [configure the runtime to be lazy](https://docs.marimo.io/guides/configuration/runtime_configuration.html),marking affected cells as stale instead of automatically running them. This ensures the consistency of program states while avoiding accidental execution of costly cells.

### Synchronized UI Elements

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="Synchronized UI Elements GIF" />

Interact with [UI elements](https://docs.marimo.io/guides/interactivity.html) like [sliders](https://docs.marimo.io/api/inputs/slider.html#slider), [dropdowns](https://docs.marimo.io/api/inputs/dropdown.html), [dataframe transformers](https://docs.marimo.io/api/inputs/dataframe.html), and [chat interfaces](https://docs.marimo.io/api/inputs/chat.html), and the cells that use them are automatically re-run with their latest values.

### Interactive Dataframes

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Interactive Dataframes GIF" />

Interact with [UI elements](https://docs.marimo.io/guides/interactivity.html) like [sliders](https://docs.marimo.io/api/inputs/slider.html#slider), [dropdowns](https://docs.marimo.io/api/inputs/dropdown.html), [dataframe transformers](https://docs.marimo.io/api/inputs/dataframe.html), and [chat interfaces](https://docs.marimo.io/api/inputs/chat.html), and the cells that use them are automatically re-run with their latest values.

### Generate with Data-Aware AI

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="Generate with Data-Aware AI GIF" />

Generate code with an AI assistant. It is highly specialized for working with data, with context about your variables in memory; [zero-shot entire notebooks](https://docs.marimo.io/guides/generate_with_ai/text_to_notebook/).

### Data Querying with SQL

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="Data Querying with SQL image" />

Build SQL queries that depend on Python values and execute them against dataframes, databases, lakehouses, CSVs, Google Sheets, or anything else using our built-in SQL engine, which returns the result as a Python dataframe.

### Other features include:
* Dynamic markdown
* Built-in package management
* Deterministic execution order
* Performant runtime
* Extensive Batteries-included features

## Quickstart

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

For recommended packages:

```bash
pip install marimo[recommended]
```

**Create and Edit Notebooks:**

```bash
marimo edit
```

**Run as Web App:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Running marimo as web app" />

**Execute as Script:**

```bash
python your_notebook.py
```

**Convert Jupyter Notebooks:**

```bash
marimo convert your_notebook.ipynb > your_notebook.py
```
Or use the [web interface](https://marimo.io/convert).

**Tutorials:**
```bash
marimo tutorial --help
```

## Learn More

marimo offers a rich environment for users of all skill levels.

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
        <img src="https://marimo.io/shield.svg" alt="marimo playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="marimo playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="marimo playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="marimo playground"/>
      </a>
    </td>
  </tr>
</table>

Explore the [docs](https://docs.marimo.io), [examples](https://docs.marimo.io/examples/), and [gallery](https://marimo.io/gallery) to get started.

## Contributing

We welcome all contributions! See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Questions?  [Join us on Discord](https://marimo.io/discord?ref=readme).

## Community

Join our growing community:

-   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
-   💬 [Chat on Discord](https://marimo.io/discord?ref=readme)
-   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
-   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
-   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
-   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
-   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
-   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
-   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS Affiliated Project.** marimo is part of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS affiliated project"/>

## Inspiration

marimo is inspired by projects like Pluto.jl and ObservableHQ, driving a movement towards reactive dataflow programming.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo logo">
</p>