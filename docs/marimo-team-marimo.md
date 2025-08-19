<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="marimo Logo">
</p>

<p align="center">
  <em>A revolutionary Python notebook experience: create reproducible, interactive, and shareable Python programs with ease.</em>
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
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

## What is marimo?

**marimo** is a reactive Python notebook that transforms how you work with code and data. It's designed to be reproducible, git-friendly, and deployable as scripts or interactive apps, replacing tools like Jupyter, Streamlit, and more. 

[**Check out the original repo on GitHub**](https://github.com/marimo-team/marimo)

## Key Features

*   🚀 **Batteries-included:** A comprehensive tool, replacing multiple tools such as `jupyter`, `streamlit`, `jupytext`, `ipywidgets`, `papermill`, and more.
*   ⚡️ **Reactive:** Cells update automatically based on dependencies, ensuring code and outputs stay consistent.  [Learn more](https://docs.marimo.io/guides/reactivity.html)
*   🖐️ **Interactive:** Easily bind UI elements (sliders, tables, plots) to Python code without callbacks.  [Explore Interactivity](https://docs.marimo.io/guides/interactivity.html)
*   🐍 **Git-Friendly:** Notebooks are stored as `.py` files, making version control simple.
*   🛢️ **Data-Focused:** Integrate data seamlessly with SQL support and powerful dataframe functionalities. [Learn about SQL in marimo](https://docs.marimo.io/guides/working_with_data/sql.html)
*   🤖 **AI-Native:** Leverage AI to generate code and accelerate data analysis.  [Generate with AI](https://docs.marimo.io/guides/generate_with_ai/)
*   🔬 **Reproducible:** Achieve consistent results with no hidden state, deterministic execution, and built-in package management.
*   🏃 **Executable:** Run notebooks as standard Python scripts with CLI argument support.  [Execute as a Script](https://docs.marimo.io/guides/scripts.html)
*   🛜 **Shareable:** Deploy notebooks as interactive web apps or slides, or run them in the browser via WASM.  [Deploy Apps](https://docs.marimo.io/guides/apps.html)
*   🧩 **Reusable:** Import functions and classes from one notebook to another for better organization.  [Reuse Functions](https://docs.marimo.io/guides/reusing_functions/)
*   🧪 **Testable:** Use pytest to test your notebooks, ensuring code quality. [Run Pytest](https://docs.marimo.io/guides/testing/)
*   ⌨️ **Modern Editor:** Benefit from a modern editor experience with GitHub Copilot, AI assistants, Vim keybindings, and a variable explorer.  [Editor Features](https://docs.marimo.io/guides/editor_features/index.html)

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [quickstart](#quickstart) for a primer on our CLI._

## A Revolutionary Reactive Programming Environment

marimo's core principle is ensuring your notebook code, outputs, and program state remain synchronized. This [addresses many issues](https://docs.marimo.io/faq.html#faq-problems) associated with traditional notebooks.

**A Reactive Programming Environment.**
Run a cell, and marimo *reacts* by automatically executing dependent cells, avoiding the need to manually re-run cells. Deleting a cell removes its variables from memory, eliminating hidden state.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Notebook">

<a name="expensive-notebooks"></a>

**Compatible with Expensive Notebooks.** Configure marimo's runtime to be lazy by [configuring the runtime](https://docs.marimo.io/guides/configuration/runtime_configuration.html), which marks affected cells as stale without automatically running them. This ensures program state while preventing accidental execution of expensive cells.

**Synchronized UI Elements.**  Interact with [UI elements](https://docs.marimo.io/guides/interactivity.html), such as [sliders](https://docs.marimo.io/api/inputs/slider.html#slider), [dropdowns](https://docs.marimo.io/api/inputs/dropdown.html), [dataframe transformers](https://docs.marimo.io/api/inputs/dataframe.html), and [chat interfaces](https://docs.marimo.io/api/inputs/chat.html), and the cells that use them are automatically re-run with their latest values.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI Interaction">

**Interactive Dataframes.** [Page through, search, filter, and sort](https://docs.marimo.io/guides/working_with_data/dataframes.html) millions of rows instantly, without writing code.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Interactive Dataframes">

**Generate Cells with Data-Aware AI.** [Generate code with an AI assistant](https://docs.marimo.io/guides/editor_features/ai_completion/) optimized for data work, with context about your variables in memory; [zero-shot entire notebooks](https://docs.marimo.io/guides/generate_with_ai/text_to_notebook/). Customize the system prompt, bring your own API keys, or use local models.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI-Powered Code Generation">

**Query Data with SQL.** Construct [SQL](https://docs.marimo.io/guides/working_with_data/sql.html) queries dependent on Python values and run them against dataframes, databases, lakehouses, CSVs, Google Sheets, and more using our built-in SQL engine, which returns the result as a Python dataframe.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Queries in marimo">

Your notebooks remain pure Python, even when incorporating SQL.

**Dynamic Markdown.** Use markdown parametrized by Python variables to tell dynamic stories that depend on Python data.

**Built-in Package Management.** marimo includes comprehensive package manager support, allowing you to [install packages on import](https://docs.marimo.io/guides/editor_features/package_management.html).  marimo can even [serialize package requirements](https://docs.marimo.io/guides/package_management/inlining_dependencies/) in notebook files and auto-install them in isolated venv sandboxes.

**Deterministic Execution Order.** Notebooks execute in a deterministic order, based on variable references, not cell positions. This allows for more organized notebook structures.

**Performant Runtime.** marimo runs only the cells that need to be executed through static analysis of your code.

**Batteries-Included.** marimo comes with many features such as GitHub Copilot, AI assistants, Ruff code formatting, HTML export, and a VS Code extension.  [See more](https://docs.marimo.io/guides/editor_features/index.html)

## Quickstart

_The [marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq)
on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an
overview of many features._

**Installation.** In your terminal, run:

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

For installing with additional dependencies for SQL cells, AI completion, etc., run:

```bash
pip install marimo[recommended]
```

**Create Notebooks.**

Create or edit notebooks with:

```bash
marimo edit
```

**Run Apps.** Run your notebook as a web app, with the Python code hidden and uneditable:

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Running as a Web App">

**Execute as Scripts.** Execute a notebook as a script at the command line:

```bash
python your_notebook.py
```

**Convert Jupyter Notebooks.** Use the CLI to automatically convert Jupyter notebooks:

```bash
marimo convert your_notebook.ipynb > your_notebook.py
```

Alternatively, use our [web interface](https://marimo.io/convert).

**Tutorials.**
List all tutorials:

```bash
marimo tutorial --help
```

**Share Cloud-Based Notebooks.** Use
[molab](https://molab.marimo.io/notebooks), a cloud-based marimo notebook service to create and share notebook links.

## Questions?

Consult the [FAQ](https://docs.marimo.io/faq.html) for answers.

## Learn More

marimo is easy to learn and use, with plenty of features for power users.
For example, here's an embedding visualizer made in marimo
([video](https://marimo.io/videos/landing/full.mp4)):

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/embedding.gif" width="700px" alt="Embedding Visualizer">

Explore our [docs](https://docs.marimo.io), [usage examples](https://docs.marimo.io/examples/), and [gallery](https://marimo.io/gallery) to delve deeper.

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive Notebook">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="UI Interaction">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Interactive DataFrames">
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
        <img src="https://marimo.io/shield.svg" alt="marimo Playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="marimo Playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="marimo Playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="marimo Playground"/>
      </a>
    </td>
  </tr>
</table>

## Contributing

We welcome contributions! You don't need to be an expert to help out.  See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Questions? Reach out [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join our growing community!

-   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
-   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
-   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
-   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
-   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
-   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
-   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
-   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
-   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS Affiliated Project.** marimo is part of the NumFOCUS community, which includes projects such as NumPy, SciPy, and Matplotlib.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS Affiliated Project" />

## Inspiration ✨

marimo is a **reinvention** of the Python notebook, transforming it into a reproducible, interactive, and shareable Python program.

We believe better tools lead to better thinking. With marimo, we strive to provide the Python community with a superior environment for research, communication, experimentation, and teaching computational science.

Our inspiration stems from many projects, including [Pluto.jl](https://github.com/fonsp/Pluto.jl), [ObservableHQ](https://observablehq.com/tutorials), and [Bret Victor's essays](http://worrydream.com/). marimo is part of the movement towards reactive dataflow programming. From [IPyflow](https://github.com/ipyflow/ipyflow), [streamlit](https://github.com/streamlit/streamlit), [TensorFlow](https://github.com/tensorflow/tensorflow), [PyTorch](https://github.com/pytorch/pytorch/tree/main), [JAX](https://github.com/google/jax), and [React](https://github.com/facebook/react), the ideas of functional, declarative, and reactive programming are revolutionizing a wide array of tools.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo Logo">
</p>