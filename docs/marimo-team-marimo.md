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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI Version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda Version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"></a>
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

## Marimo: The Reactive Python Notebook for Data Science

**Marimo** is a revolutionary reactive Python notebook that transforms how you work with code, data, and interactive applications.  Build reproducible, git-friendly, and deployable Python programs with ease.  [Learn more on GitHub](https://github.com/marimo-team/marimo).

**Key Features:**

*   🚀 **Batteries-included:** Replaces Jupyter, Streamlit, and more.
*   ⚡️ **Reactive:** Automatically updates dependent cells.
*   🖐️ **Interactive:** Bind UI elements (sliders, plots, etc.) directly to Python.
*   🐍 **Git-friendly:** Stored as `.py` files for easy version control.
*   🛢️ **Designed for Data:** Integrate SQL, filter dataframes, and more.
*   🤖 **AI-native:** Generate cells and notebooks with AI assistants.
*   🔬 **Reproducible:** Ensures consistent code execution and state.
*   🏃 **Executable:** Run notebooks as standalone Python scripts.
*   🛜 **Shareable:** Deploy as interactive web apps or slides.
*   🧩 **Reusable:** Import functions and classes from other notebooks.
*   🧪 **Testable:** Seamlessly integrates with pytest for testing.
*   ⌨️ **Modern Editor:** Features like GitHub Copilot, AI assistants, and more.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz)!_

_See the [Quickstart](#quickstart) below for CLI commands._

## A Deeper Dive into Marimo's Features

Marimo is a game-changer for data science, offering a suite of features designed to simplify your workflow and enhance collaboration.

**Reactive Programming Environment:**  When you run a cell, marimo intelligently updates dependent cells, eliminating the need for manual re-runs.  Deleting a cell removes its variables from memory, ensuring a clean and predictable state.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Notebook Example" />

<a name="expensive-notebooks"></a>

**Optimized for Expensive Notebooks:**  Configure marimo to be lazy, marking cells as stale instead of running them automatically.  This protects you from unintended and expensive computations while ensuring your code's consistency.

**Synchronized UI Elements:**  Integrate interactive UI elements (sliders, dropdowns, etc.) and see the results reflected immediately.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="Interactive UI elements" />

**Interactive DataFrames:**  Explore and interact with large datasets with intuitive features like pagination, searching, filtering, and sorting.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Interactive DataFrames" />

**AI-Powered Code Generation:**  Utilize AI assistants for data-specific code generation, allowing you to generate code and notebooks with minimal input.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI-powered code generation" />

**SQL Integration:**  Write and execute SQL queries directly within your notebooks, supporting connections to dataframes, databases, and more.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Integration" />

**Dynamic Markdown:** Create dynamic markdown content using variables.

**Built-in Package Management:** Install packages on import, with dependency serialization and automated venv management.

**Deterministic Execution:**  Code executes in a deterministic order based on variable references, allowing you to organize your notebooks intuitively.

**Performant Runtime:**  Marimo optimizes performance by running only the necessary cells.

**Comprehensive Tooling:** Enjoy a rich development environment with features like GitHub Copilot, a VS Code extension, and more.

## Quickstart

_Explore the [marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq) to learn more._

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

For extended functionality, install with:

```bash
pip install marimo[recommended]
```

**Create and Edit Notebooks:**

```bash
marimo edit
```

**Run Apps:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="marimo app example" />

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

## FAQs and Support

Find answers to common questions in the [FAQ](https://docs.marimo.io/faq.html).

## Learn More

Dive deeper into marimo's capabilities with the resources below:

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive Programming Tutorial" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Input elements" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Plots" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Layout examples" />
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
        <img src="https://marimo.io/shield.svg" alt="Marimo playground link"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Marimo app example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Marimo app example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Marimo app example"/>
      </a>
    </td>
  </tr>
</table>

*   **Docs:** [https://docs.marimo.io](https://docs.marimo.io)
*   **Examples:** [https://docs.marimo.io/examples/](https://docs.marimo.io/examples/)
*   **Gallery:** [https://marimo.io/gallery](https://marimo.io/gallery)

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Need help?  Connect with us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join the marimo community and stay connected:

*   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**NumFOCUS Affiliated:** marimo is a member of the NumFOCUS community, supporting projects like NumPy and SciPy.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS logo" />

## Inspiration ✨

marimo is a **reinvention** of the Python notebook as a reproducible, interactive,
and shareable Python program. We aim to provide better tools for better minds.

Our inspiration comes from [Pluto.jl](https://github.com/fonsp/Pluto.jl),
[ObservableHQ](https://observablehq.com/tutorials), and
[Bret Victor's essays](http://worrydream.com/). Marimo is part of
a movement towards reactive dataflow programming.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="Marimo logo">
</p>