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

# marimo: The Reactive Python Notebook for Data Science and App Development

**marimo** is a revolutionary reactive Python notebook that transforms the way you work with data, offering reproducibility, interactivity, and seamless deployment – [learn more on GitHub](https://github.com/marimo-team/marimo).

## Key Features

*   🚀 **Batteries-included:** Replaces Jupyter, Streamlit, and more, providing a comprehensive data science environment.
*   ⚡️ **Reactive:** Automatically updates dependent cells when you change a cell, ensuring consistency and eliminating errors.
*   🖐️ **Interactive:** Easily create interactive UIs with sliders, tables, plots, and more, without complex callbacks.
*   🐍 **Git-Friendly:** Notebooks are stored as `.py` files, making them version-control-friendly.
*   🛢️ **Data-Focused:** Native SQL support for querying dataframes, databases, and data warehouses.
*   🤖 **AI-Native:** Generate code cells with an AI assistant tailored for data work.
*   🔬 **Reproducible:** Deterministic execution and built-in package management ensure your work is always replicable.
*   🏃 **Executable:** Run notebooks as Python scripts, parameterized by CLI arguments.
*   🛜 **Shareable:** Deploy your notebooks as interactive web apps or slides, and even run them in the browser via WASM.
*   🧩 **Reusable:** Import functions and classes from one notebook to another for modularity.
*   🧪 **Testable:** Easily run pytest on your notebooks.
*   ⌨️ **Modern Editor:** Benefit from GitHub Copilot integration, AI assistants, Vim keybindings, and a variable explorer.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [quickstart](#quickstart) for a primer on our CLI._

## How marimo Works

marimo offers a truly reactive programming environment, guaranteeing consistent code, outputs, and state. It eliminates the common issues found in traditional notebooks.

### Reactive Programming Environment

Run a cell, and marimo automatically runs cells that depend on it. Delete a cell and marimo scrubs variables from memory, preventing hidden state.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive demonstration" />

### Addressing Expensive Notebooks

marimo lets you configure the runtime to be lazy, marking cells as stale instead of running them automatically. This prevents unintended execution of time-consuming cells.

### Synchronized UI Elements

Interactive UI elements like sliders, dropdowns, and dataframe transformers automatically update related cells with their latest values.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI elements demonstration" />

### Interactive DataFrames

Quickly page through, search, filter, and sort millions of rows without any coding required.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Interactive DataFrames demonstration" />

### AI-Powered Code Generation

Use an AI assistant specialized for data work, generating code with context about your variables in memory.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI code generation demonstration" />

### SQL Integration

Build SQL queries that depend on Python values and execute them against various data sources.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Integration demonstration" />

### Dynamic Markdown

Use markdown parametrized by Python variables.

### Package Management

Built-in support for all major package managers, including the ability to install packages on import and serialize requirements.

### Deterministic Execution Order

Notebooks are executed in a deterministic order, based on variable references.

### Performant Runtime

marimo runs only the necessary cells by statically analyzing your code.

### Batteries-Included

marimo comes with GitHub Copilot, AI assistants, Ruff code formatting, HTML export, fast code completion, a VS Code extension, an interactive dataframe viewer, and many other features.

## Quickstart

_The [marimo concepts
playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq)
on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an
overview of many features._

**Installation.** In a terminal, run

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with additional dependencies that unlock SQL cells, AI completion, and more,
run

```bash
pip install marimo[recommended]
```

**Create notebooks.**

Create or edit notebooks with

```bash
marimo edit
```

**Run apps.** Run your notebook as a web app, with Python
code hidden and uneditable:

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" />

**Execute as scripts.** Execute a notebook as a script at the
command line:

```bash
python your_notebook.py
```

**Automatically convert Jupyter notebooks.** Automatically convert Jupyter
notebooks to marimo notebooks with the CLI

```bash
marimo convert your_notebook.ipynb > your_notebook.py
```

or use our [web interface](https://marimo.io/convert).

**Tutorials.**
List all tutorials:

```bash
marimo tutorial --help
```

**Share cloud-based notebooks.** Use
[molab](https://molab.marimo.io/notebooks), a cloud-based marimo notebook
service similar to Google Colab, to create and share notebook links.

## Get Help

See the [FAQ](https://docs.marimo.io/faq.html) at our docs.

## Learn More

marimo is easy to get started with, with lots of room for power users.
For example, here's an embedding visualizer made in marimo
([video](https://marimo.io/videos/landing/full.mp4)):

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/embedding.gif" width="700px" />

Explore our [docs](https://docs.marimo.io), [usage examples](https://docs.marimo.io/examples/), and [gallery](https://marimo.io/gallery) to discover the full potential of marimo.

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" />
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
        <img src="https://marimo.io/shield.svg"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg"/>
      </a>
    </td>
  </tr>
</table>

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Questions? Reach out to us [on Discord](https://marimo.io/discord?ref=readme).

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

**A NumFOCUS affiliated project.** marimo is a member of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS Affiliated Project" />

## Inspiration ✨

marimo is inspired by [Pluto.jl](https://github.com/fonsp/Pluto.jl), [ObservableHQ](https://observablehq.com/tutorials), and [Bret Victor's essays](http://worrydream.com/) and aims to provide a better programming environment for research, communication, and education.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo logotype horizontal">
</p>
```

Key improvements and SEO considerations:

*   **Clear, Concise Title:** The title includes relevant keywords ("Reactive Python Notebook," "Data Science," "App Development") to improve search engine visibility.
*   **Strong Hook:** The introductory sentence immediately highlights the core value proposition.
*   **Well-Structured Headings:** Uses `##` for clear organization, improving readability and SEO.
*   **Keyword Optimization:** Includes relevant keywords throughout the content (e.g., "reactive," "Python notebook," "data science," "app development," "interactive").
*   **Bulleted Key Features:** Uses bullet points to make the key benefits easy to scan.
*   **Emphasis on Benefits:**  Focuses on what marimo *does* for the user.
*   **Concise Descriptions:** Avoids overly verbose descriptions.
*   **Call to Action (Install/Playground):** Encourages user engagement.
*   **Internal Linking:** Uses internal links to other sections for easy navigation.
*   **Alt Text for Images:** Added `alt` text to the images for accessibility and SEO.
*   **Community and Social Links:**  Provides a comprehensive list of ways to connect with the marimo community.
*   **NumFOCUS Affiliation:** Highlights the project's connection to a respected organization.
*   **Clear Inspiration Section:**  Clearly states the inspiration and broader movement.
*   **Markdown Formatting:** Uses proper markdown formatting for consistent display.
*   **Concise & Informative:** Rephrased sentences for better readability.
*   **Includes Important Sections:** Retains and improves the Quickstart, Get Help, Learn More, Contributing, and Community sections.