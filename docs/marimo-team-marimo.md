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
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Chinese.md" target="_blank"><b>简体中文</b></a>
  <b> | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Japanese.md" target="_blank"><b>日本語</b></a>
  <b> | </b>
  <a href="https://github.com/marimo-team/marimo/blob/main/README_Spanish.md" target="_blank"><b>Español</b></a>
</p>

<p align="center">
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="discord" /></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

## **marimo: The Reactive Python Notebook for Modern Data Work**

Tired of the limitations of traditional notebooks?  marimo is a revolutionary reactive Python notebook designed for reproducibility, ease of use, and deployment, offering a powerful and interactive environment for data science and Python development.  [Check out the original repo](https://github.com/marimo-team/marimo).

**Key Features:**

*   **🚀 Batteries-Included:** Replaces `jupyter`, `streamlit`, `jupytext`, `ipywidgets`, `papermill`, and more, streamlining your workflow.
*   **⚡️ Reactive Execution:**  Automatically updates dependent cells when a cell's code or input changes, ensuring consistency and eliminating manual re-runs.
*   **🖐️ Interactive UI Elements:** Easily bind sliders, tables, plots, and other UI elements directly to Python code, enabling dynamic and engaging data exploration – no callbacks required.
*   **🐍 Git-Friendly:**  marimo notebooks are stored as plain `.py` files, making version control and collaboration seamless.
*   **🛢️ Data-Centric Design:**  Built-in support for querying dataframes, databases, warehouses, or lakehouses with SQL. Easily filter, search and analyze dataframes.
*   **🤖 AI-Powered Development:**  Leverage AI to generate cells tailored for data work.
*   **🔬 Reproducible Results:**  Guarantee consistent results with no hidden state, deterministic execution, and built-in package management.
*   **🏃 Executable Scripts:**  Run notebooks as Python scripts, complete with CLI argument support.
*   **🛜 Shareable & Deployable:** Easily deploy notebooks as interactive web apps or slides, and even run them in the browser via WASM.
*   **🧩 Reusable Code:**  Import and reuse functions and classes between notebooks.
*   **🧪 Testable Notebooks:** Integrate testing into your workflow with support for running pytest directly on notebooks.
*   **⌨️ Modern Editor:**  Benefit from a feature-rich editor with GitHub Copilot, AI assistants, Vim keybindings, a variable explorer, and more.

**Get Started:**

```bash
pip install marimo && marimo tutorial intro
```

_Explore marimo in your browser at [our online playground](https://marimo.app/l/c7h6pz)!_

_Jump to the [quickstart](#quickstart) to dive in with the CLI._

## Reactive Programming Environment

marimo's reactive design ensures that your notebook code, outputs, and program state always remain consistent.

**A Reactive Programming Environment:** Run a cell, and marimo *reacts* by automatically running cells that depend on its variables. Delete a cell, and marimo scrubs its variables from memory, eliminating hidden state.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="marimo reactivity demo">

**Compatible with Expensive Notebooks:**  Configure the runtime to be lazy, marking affected cells as stale instead of automatically executing them.

**Synchronized UI Elements:**  Interactive elements like sliders, dropdowns, and chat interfaces automatically trigger re-runs of dependent cells with the latest values.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="marimo UI demo">

**Interactive Dataframes:**  Quickly page through, search, filter, and sort millions of rows.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="marimo dataframe demo">

**Generate Cells with Data-Aware AI:**  Create code with an AI assistant, highly specialized for working with data and your in-memory variables.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="marimo AI demo">

**Query Data with SQL:**  Build SQL queries that depend on Python values, and execute them against dataframes, databases, lakehouses, CSVs, and Google Sheets using the built-in SQL engine.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="marimo SQL demo">

## Quickstart

_Find an overview of marimo's features in the [marimo concepts
playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq) on our [YouTube channel](https://www.youtube.com/@marimo-team)._

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with additional dependencies that unlock SQL cells, AI completion, and more, run:

```bash
pip install marimo[recommended]
```

**Create and Edit Notebooks:**

```bash
marimo edit
```

**Run as Web Apps:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="marimo web app demo"/>

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

##  Need Help?

*   Check out the [FAQ](https://docs.marimo.io/faq.html) in our documentation.

## Learn More

marimo is designed to be easy to use, with advanced features for power users.

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive concepts">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Interactive inputs">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Plotting data">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Custom layouts">
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
        <img src="https://marimo.io/shield.svg" alt="Browser demo 1"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Browser demo 2"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Browser demo 3"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Browser demo 4"/>
      </a>
    </td>
  </tr>
</table>

## Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

>  For questions, please reach out [on Discord](https://marimo.io/discord?ref=readme).

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

**A NumFOCUS affiliated project.** marimo is part of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFocus logo" />

## Inspiration ✨

marimo reimagines the Python notebook as a reproducible, interactive, and shareable Python program.

We believe better tools lead to better thinking, and with marimo, we aim to provide the Python community a better environment for research, communication, experimentation, and learning.

Inspired by [Pluto.jl](https://github.com/fonsp/Pluto.jl),
[ObservableHQ](https://observablehq.com/tutorials), [Bret Victor's essays](http://worrydream.com/), and the broader movement toward reactive dataflow programming, including projects like [IPyflow](https://github.com/ipyflow/ipyflow), [streamlit](https://github.com/streamlit/streamlit), [TensorFlow](https://github.com/tensorflow/tensorflow), [PyTorch](https://github.com/pytorch/pytorch/tree/main), [JAX](https://github.com/google/jax), and [React](https://github.com/facebook/react).

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo logo">
</p>