<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="Marimo Logo">
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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI version"/></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda version"/></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"/></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"/></a>
</p>

## Marimo: The Reactive Python Notebook for Data Science

**Marimo** is a next-generation reactive Python notebook environment that transforms how you work with code, data, and AI, making your projects reproducible, shareable, and incredibly efficient.  ([Check out the original repo](https://github.com/marimo-team/marimo))

**Key Features:**

*   **Reactive Programming:**  Automatically updates dependent cells when you run a cell or interact with a UI element, ensuring consistency.
*   **Interactive Elements:**  Easily integrate interactive sliders, dropdowns, dataframes, and more directly into your notebooks without complex callback functions.
*   **Pure Python:**  marimo notebooks are stored as `.py` files, making them Git-friendly, executable as scripts, and easy to integrate into your existing workflow.
*   **Data-Focused:**  Seamlessly work with data using SQL, filtering and searching dataframes, and connecting to databases.
*   **AI-Powered:** Generate code using AI assistants tailored for data science tasks, including zero-shot notebook generation.
*   **Reproducible & Testable:** Built-in package management, deterministic execution, and pytest integration make your work reliable and easy to test.
*   **Executable & Shareable:** Run your notebooks as Python scripts, deploy them as interactive web apps, or create presentation-ready slides.
*   **Modern Editor:** Features GitHub Copilot, AI assistants, Vim keybindings, a variable explorer, and other enhancements for a superior coding experience.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

## Core Capabilities

marimo revolutionizes your workflow by guaranteeing consistency between code, outputs, and program state. This eliminates common issues found in traditional notebooks, ensuring reliability and efficiency.

**A Reactive Programming Environment:** When you run a cell, marimo intelligently re-runs dependent cells to reflect changes automatically, minimizing manual intervention and errors.  Deleting a cell removes its variables from memory, preventing hidden state issues.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Notebook">

**Supports Complex Workflows:** Configure the runtime to mark expensive cells as stale instead of automatically running them, preserving program state while preventing accidental resource consumption.

**Interactive & Dynamic:** Integrate UI elements such as sliders, dropdowns, and dataframes that instantly update the cells that use them.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="Interactive UI Elements">

**Fast Data Exploration:** Effortlessly page through, search, filter, and sort millions of rows within your dataframes.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Interactive Dataframe">

**AI-Enhanced Development:** Generate code for data-related tasks using an AI assistant that understands your variables and can create entire notebooks from text prompts.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI Code Generation">

**Built-In SQL Engine:** Easily create SQL queries using a built-in SQL engine and integrate them with Python variables.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Queries">

## Getting Started

### Quickstart

_The [marimo concepts
playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq)
on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an
overview of many features._

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with additional dependencies that unlock SQL cells, AI completion, and more,
run

```bash
pip install marimo[recommended]
```

**Create & Edit Notebooks:**

```bash
marimo edit
```

**Run as Web Apps:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Run as Web App"/>

**Execute as Scripts:**

```bash
python your_notebook.py
```

**Convert Jupyter Notebooks:**

```bash
marimo convert your_notebook.ipynb > your_notebook.py
```

or use our [web interface](https://marimo.io/convert).

**Tutorials:**

```bash
marimo tutorial --help
```

**Share cloud-based notebooks:** Use
[molab](https://molab.marimo.io/notebooks), a cloud-based marimo notebook
service similar to Google Colab, to create and share notebook links.

## Resources & Support

*   **[Documentation](https://docs.marimo.io):** Comprehensive guides and API references.
*   **[Examples](https://docs.marimo.io/examples/):** Explore practical applications.
*   **[Gallery](https://marimo.io/gallery):** View a showcase of marimo projects.
*   **[FAQ](https://docs.marimo.io/faq.html):** Find answers to common questions.
*   **[Discord](https://marimo.io/discord?ref=readme):** Get help from the community.

## Learn More

marimo is powerful and easy to use, with many options for advanced users.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/embedding.gif" width="700px" alt="Embedding Visualizer">

**See the following for more info:**

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
        <img src="https://marimo.io/shield.svg" alt="Marimo Playground"/>
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

We welcome contributions! For guidance on how to contribute, please see [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md).

> Questions? Reach out to us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join our growing community!

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

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS">

## Inspiration

marimo rethinks the Python notebook, providing a reproducible, interactive, and shareable Python program, rather than a problematic JSON scratchpad.

Our inspiration comes from projects such as [Pluto.jl](https://github.com/fonsp/Pluto.jl), [ObservableHQ](https://observablehq.com/tutorials), and [Bret Victor's essays](http://worrydream.com/).

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="Marimo Logo">
</p>