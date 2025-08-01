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

# marimo: The Reactive Python Notebook for Data Science and Beyond

marimo is a modern, reactive Python notebook that reimagines how you write, share, and deploy your code and data. [Learn more about marimo](https://github.com/marimo-team/marimo).

**Key Features:**

*   **🚀 Batteries-Included:** Replaces tools like Jupyter, Streamlit, and more, streamlining your workflow.
*   **⚡ Reactive Programming:**  Automatic updates: run a cell, and marimo smartly re-runs dependent cells, keeping your code and outputs synchronized.
*   **🖐️ Interactive UIs:**  Easily bind UI elements (sliders, tables, plots) to your Python code for dynamic explorations.
*   **🐍 Git-Friendly:**  Notebooks are stored as pure Python files, making version control simple.
*   **🛢️ Data-First Design:**  Seamlessly query dataframes and databases with SQL, and filter and search datasets with ease.
*   **🤖 AI-Powered Development:**  Generate cells with AI tailored for data work.
*   **🔬 Reproducible & Reliable:**  Guaranteed deterministic execution, no hidden state, and built-in package management.
*   **🏃 Executable as Scripts:**  Run notebooks as standard Python scripts with command-line arguments.
*   **🛜 Shareable & Deployable:**  Deploy your notebooks as interactive web apps or slides, and even run them in the browser.
*   **🧩 Reusable Code:**  Import functions and classes between notebooks for modularity.
*   **🧪 Testable Notebooks:** Integrate with pytest for robust testing.
*   **⌨️ Modern Editor:**  Benefit from a modern editor with features like GitHub Copilot, AI assistants, and more for a great coding experience.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

## Core Concepts

marimo is built on the principle of reactivity, ensuring that your code, outputs, and program state always remain consistent. This approach eliminates many of the common issues found in traditional notebooks like Jupyter.

*   **Reactive Execution:**  When a cell is run, marimo intelligently executes all dependent cells, guaranteeing that your program's state is up-to-date.
*   **UI Synchronization:**  Interactive UI elements like sliders and dropdowns are automatically synchronized with your Python code, allowing for seamless interaction and exploration.
*   **Dataframe Interaction:**  Effortlessly page through, search, filter, and sort large datasets directly within the notebook.
*   **AI-Powered Code Generation:** Generate code with data-aware AI assistants tailored for working with data, with context about your variables in memory.
*   **SQL Integration:** Build SQL queries that depend on Python variables and execute them against dataframes, databases, and more.
*   **Dynamic Markdown:** Create dynamic markdown, powered by Python variables, for creating narratives that update as your data does.
*   **Package Management:** Built-in package management facilitates easy installation and management.

## Getting Started

### Installation

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with recommended dependencies (SQL, AI, etc.):

```bash
pip install marimo[recommended]
```

### Basic Usage

*   **Create/Edit Notebooks:**
    ```bash
    marimo edit
    ```
*   **Run as Web App:**
    ```bash
    marimo run your_notebook.py
    ```
*   **Execute as Script:**
    ```bash
    python your_notebook.py
    ```
*   **Convert Jupyter Notebooks:**
    ```bash
    marimo convert your_notebook.ipynb > your_notebook.py
    ```

## Learn More

Dive deeper into marimo's features and capabilities:

*   [marimo Docs](https://docs.marimo.io)
*   [Usage Examples](https://docs.marimo.io/examples/)
*   [Gallery](https://marimo.io/gallery)

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

## Community

Join the marimo community:

*   [Star on GitHub](https://github.com/marimo-team/marimo)
*   [Join our Discord](https://marimo.io/discord?ref=readme)
*   [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   [Join our Cloud Waitlist](https://marimo.io/cloud)
*   [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   [Follow on Bluesky](https://bsky.app/profile/marimo.io)
*   [Follow on Twitter](https://twitter.com/marimo_io)
*   [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   [Follow on LinkedIn](https://www.linkedin.com/company/marimo-io)

**Affiliated with NumFOCUS.** marimo is a NumFOCUS affiliated project.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" />