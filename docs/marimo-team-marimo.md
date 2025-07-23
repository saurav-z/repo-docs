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


# marimo: The Reactive Python Notebook for Data Science and More

marimo is a revolutionary reactive Python notebook that empowers data scientists and developers to create reproducible, interactive, and shareable Python programs; **transforming the way you work with data and build applications.** ([See the original repo](https://github.com/marimo-team/marimo)).

## Key Features

*   🚀 **Batteries-Included:** Replaces tools like Jupyter, Streamlit, and more, offering a comprehensive environment.
*   ⚡️ **Reactive Execution:**  Automatically updates dependent cells when a cell's code or data changes, ensuring consistency.
*   🖐️ **Interactive UI Elements:** Seamlessly bind sliders, tables, plots, and other UI elements to Python code without callbacks.
*   🐍 **Git-Friendly:**  marimo notebooks are stored as plain `.py` files, making version control straightforward.
*   🛢️ **Data-Focused:** Query dataframes, databases, and data warehouses with SQL, and easily filter and search dataframes.
*   🤖 **AI-Enhanced:** Generate code with AI assistants tailored for data work.
*   🔬 **Reproducible:** Guarantees no hidden state and deterministic execution, with built-in package management.
*   🏃 **Executable Scripts:** Run notebooks as standard Python scripts, parameterized by CLI arguments.
*   🛜 **Shareable Apps:** Deploy notebooks as interactive web apps or create slide presentations, and even run them in the browser via WASM.
*   🧩 **Reusable Code:** Import functions and classes between notebooks for better code organization.
*   🧪 **Testable:** Integrate pytest for robust testing of your notebooks.
*   ⌨️ **Modern Editor:** Benefit from a modern editor with features like GitHub Copilot integration, AI assistants, and a variable explorer.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

## A Reactive Programming Environment

marimo guarantees consistency between your code, outputs, and program state.

**Reactive Environment:** Run a cell, and marimo automatically executes dependent cells, eliminating manual re-runs and hidden state issues.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive execution GIF" />

**Compatible with Expensive Notebooks:** Configure the runtime to be lazy, marking affected cells as stale instead of automatically running them, preventing the accidental execution of expensive cells.

**Synchronized UI Elements:** Interact with sliders, dropdowns, dataframes, and chat interfaces, and have the cells that use them automatically re-run with their latest values.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI elements GIF" />

**Interactive DataFrames:** Page through, search, filter, and sort millions of rows blazingly fast, no code required.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Dataframe GIF" />

**AI-Powered Code Generation:** Generate code with a data-aware AI assistant, customized for data work, and even create entire notebooks from text.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI Code generation GIF" />

**SQL Integration:** Build SQL queries that depend on Python values and execute them against various data sources.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Cell screenshot" />

## Quickstart

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with recommended dependencies:

```bash
pip install marimo[recommended]
```

**Create Notebooks:**

```bash
marimo edit
```

**Run as Web Apps:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Web app example GIF" />

**Execute as Scripts:**

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

Explore the capabilities of marimo through our [docs](https://docs.marimo.io), [usage examples](https://docs.marimo.io/examples/), and [gallery](https://marimo.io/gallery).

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive tutorial thumbnail" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Inputs tutorial thumbnail" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Plotting tutorial thumbnail" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Layout tutorial thumbnail" />
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
        <img src="https://marimo.io/shield.svg" alt="Playground link"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="App link"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="App link"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="App link"/>
      </a>
    </td>
  </tr>
</table>

## Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for guidelines.

> Questions?  Join us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join our community:

-   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
-   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
-   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
-   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
-   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
-   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
-   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
-   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
-   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.** marimo is part of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS logo" />

## Inspiration

marimo is a reimagining of the Python notebook, providing a more reproducible, interactive, and shareable experience. We are inspired by projects such as [Pluto.jl](https://github.com/fonsp/Pluto.jl), [ObservableHQ](https://observablehq.com/tutorials), and the work of [Bret Victor](http://worrydream.com/).  marimo is a part of a broader trend towards reactive dataflow programming.