<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="marimo Logo">
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
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

# marimo: The Reactive Python Notebook for Data Science

**Tired of Jupyter notebooks?  marimo is a modern, reactive Python notebook that transforms data analysis and app creation with reproducibility, git-friendliness, and deployability.** ([See the original repo](https://github.com/marimo-team/marimo))

## Key Features

*   🚀 **Batteries-Included:** Replaces Jupyter, Streamlit, and more, offering a comprehensive environment.
*   ⚡️ **Reactive Execution:**  Automatically runs dependent cells when a cell is updated, or marks them as stale, ensuring data consistency.
*   🖐️ **Interactive UI:**  Bind sliders, tables, plots, and more to Python code without needing callbacks.
*   🐍 **Git-Friendly:**  Notebooks are stored as standard `.py` files for easy version control.
*   🛢️ **Data-Focused:**  Query data with SQL, filter and search dataframes, and more, designed specifically for data-driven tasks.
*   🤖 **AI-Powered:**  Generate code with AI assistants tailored for data work, including zero-shot notebook generation.
*   🔬 **Reproducible:**  Eliminates hidden state with deterministic execution and built-in package management.
*   🏃 **Executable:**  Run notebooks as Python scripts, parameterized by command-line arguments.
*   🛜 **Shareable:** Deploy notebooks as interactive web apps or slides, and even run them in the browser via WASM.
*   🧩 **Reusable:**  Import functions and classes from one notebook into another.
*   🧪 **Testable:**  Integrate with pytest for robust testing of your notebooks.
*   ⌨️ **Modern Editor:** Includes features like GitHub Copilot, AI assistants, Vim keybindings, and a variable explorer.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz) which runs entirely in the browser!_

## Core Concepts

marimo guarantees that your notebook code, outputs, and program state are consistent.

*   **Reactive Programming Environment:** Run a cell and marimo automatically runs dependent cells or marks them as stale, ensuring that your code is up-to-date.
*   **Compatible with Expensive Notebooks:** Configure marimo to be lazy and prevent accidental execution of expensive cells.
*   **Synchronized UI Elements:** UI elements are automatically rerun with their latest values when interacted with.
*   **Interactive Dataframes:** Page through, search, filter, and sort millions of rows of data with no code required.
*   **Generate Cells with AI:** Leverage data-aware AI for code generation, customization, and zero-shot entire notebooks.
*   **Query Data with SQL:** Build SQL queries that depend on Python values and execute them against dataframes, databases, lakehouses, CSVs, Google Sheets, etc.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive execution">

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="Interactive UI elements">

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Interactive Dataframes">

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI Code Generation">

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL integration">

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

**Commands:**

*   Create or edit notebooks: `marimo edit`
*   Run as a web app: `marimo run your_notebook.py`
*   Execute as a script: `python your_notebook.py`
*   Convert Jupyter notebooks: `marimo convert your_notebook.ipynb > your_notebook.py` or use the [web interface](https://marimo.io/convert).
*   List tutorials: `marimo tutorial --help`
*   Cloud-based notebooks: [molab](https://molab.marimo.io/notebooks)

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="marimo application example">

## Learn More

marimo is designed for both beginners and power users.

Check out our [docs](https://docs.marimo.io), [usage examples](https://docs.marimo.io/examples/), and our [gallery](https://marimo.io/gallery) to learn more.

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Tutorial"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Inputs"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Plots"/>
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
        <img src="https://marimo.io/shield.svg" alt="Playground"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Example"/>
      </a>
    </td>
  </tr>
</table>

## Contributing

Contributions are welcome!  See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Questions? Reach out to us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join the marimo community:

*   🌟 [Star on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Discord](https://marimo.io/discord?ref=readme)
*   📧 [Newsletter](https://marimo.io/newsletter)
*   ☁️ [Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [GitHub Discussions](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Twitter](https://twitter.com/marimo_io)
*   🎥 [YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.**

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS">

## Inspiration

marimo reimagines Python notebooks as reproducible, interactive, and shareable programs.

We believe that better tools lead to better thinking and aim to provide the Python community with a superior environment for research, communication, and learning.