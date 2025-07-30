<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="marimo Logo">
</p>

<p align="center">
  <em>Transform your Python workflow with marimo: the reactive notebook for reproducibility, collaboration, and deployment.</em>
</p>

<p align="center">
  <a href="https://docs.marimo.io" target="_blank"><strong>Docs</strong></a> ·
  <a href="https://marimo.io/discord?ref=readme" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://docs.marimo.io/examples/" target="_blank"><strong>Examples</strong></a> ·
  <a href="https://marimo.io/gallery/" target="_blank"><strong>Gallery</strong></a> ·
  <a href="https://www.youtube.com/@marimo-team/" target="_blank"><strong>YouTube</strong></a>
  · <a href="https://github.com/marimo-team/marimo"><strong>GitHub</strong></a>
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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI Version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda Version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads" alt="PyPI Downloads">
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" alt="Conda Downloads">
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

##  marimo: The Reactive Python Notebook 

**marimo** is a cutting-edge reactive Python notebook designed for modern data science workflows.  It offers a powerful, reproducible, and shareable environment for exploring data, building interactive applications, and communicating insights.

### Key Features:

*   🚀 **Batteries-Included:** Replaces multiple tools like Jupyter, Streamlit, and more.
*   ⚡️ **Reactive:** Automatically updates dependent cells when a cell's input changes.
*   🖐️ **Interactive:** Easily create interactive elements like sliders, tables, and plots.
*   🐍 **Git-Friendly:** Notebooks are stored as `.py` files, enabling seamless version control.
*   🛢️ **Data-Focused:** Seamlessly query and manipulate data with SQL, dataframes, and more.
*   🤖 **AI-Native:** Leverage AI to generate and enhance your code.
*   🔬 **Reproducible:** Ensures consistent results with no hidden state and deterministic execution.
*   🏃 **Executable:** Run notebooks as standalone Python scripts.
*   🛜 **Shareable:** Deploy your work as interactive web apps or presentations.
*   🧩 **Reusable:** Import and reuse functions and classes across notebooks.
*   🧪 **Testable:**  Integrates with `pytest` for robust testing.
*   ⌨️ **Modern Editor:** Features GitHub Copilot, AI assistants, and other editor enhancements.

```python
pip install marimo && marimo tutorial intro
```

_Explore marimo in the [online playground](https://marimo.app/l/c7h6pz) or jump to the [quickstart](#quickstart) for a guide to the CLI._

## A Deep Dive into Reactive Programming

marimo redefines the notebook experience by ensuring consistency between your code, outputs, and program state.  This innovative approach addresses common challenges associated with traditional notebooks, fostering a more reliable and efficient workflow.

### Core Concepts:

*   **Reactive Execution:**  When you run a cell, marimo intelligently re-runs only the cells that depend on its output.  This eliminates manual re-runs and reduces errors.

    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Example">

*   **Handling Expensive Computations:**  Configure marimo to be "lazy" to prevent accidental execution of time-consuming cells.  This still guarantees program state consistency.

*   **Synchronized UI Elements:**  Interactive elements like sliders and dropdowns are directly linked to your Python code.  Change an input, and the relevant cells automatically update.

    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="Interactive UI Example">

*   **Interactive DataFrames:**  Effortlessly page through, search, filter, and sort massive datasets directly within your notebook.

    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Interactive DataFrame Example">

*   **AI-Powered Code Generation:**  Utilize an AI assistant to generate code tailored for data analysis, incorporating context from your variables.

    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI Code Generation Example">

*   **SQL Integration:**  Build dynamic SQL queries that integrate with your Python variables and execute against various data sources.

    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Cell Example">

*   **Dynamic Markdown:**  Create dynamic narratives using markdown that's parameterized by your Python data.

*   **Package Management:**  Easily manage dependencies, including installing packages directly from within your notebook.

*   **Deterministic Execution Order:**  Notebooks run based on variable dependencies, not cell order.

*   **Performance:** Execute only the cells that need to be run.

*   **Comprehensive Toolkit:** Built-in features include GitHub Copilot integration, a VS Code extension, and an interactive DataFrame viewer.

## Quickstart

_Explore the [marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq) on YouTube for an overview of features._

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

For extra features (SQL, AI completion):

```bash
pip install marimo[recommended]
```

**Notebook Creation & Editing:**

```bash
marimo edit
```

**Running Apps:**

```bash
marimo run your_notebook.py
```
<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Running a marimo app">

**Script Execution:**

```bash
python your_notebook.py
```

**Jupyter Notebook Conversion:**

```bash
marimo convert your_notebook.ipynb > your_notebook.py
```
or use our [web interface](https://marimo.io/convert).

**Tutorials:**

```bash
marimo tutorial --help
```

**Cloud Notebooks:** Use
[molab](https://molab.marimo.io/notebooks).

##  Need Help?

Check the [FAQ](https://docs.marimo.io/faq.html).

##  Learn More

marimo offers a versatile platform for both beginners and advanced users.

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive Concept">
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
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Layouts">
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
        <img src="https://marimo.io/shield.svg" alt="Example Playground">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Example Playground">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Example Playground">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Example Playground">
      </a>
    </td>
  </tr>
</table>

## Contributing

We welcome all contributions! Check out [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md).

> Questions? [Join us on Discord](https://marimo.io/discord?ref=readme).

## Join the Community

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

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFocus affiliated project" />

##  Inspiration

marimo reimagines the Python notebook as a reproducible, interactive program.

Inspired by projects like [Pluto.jl](https://github.com/fonsp/Pluto.jl) and [ObservableHQ](https://observablehq.com/tutorials), marimo contributes to the movement toward reactive dataflow programming.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo Logo">
</p>