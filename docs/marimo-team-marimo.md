<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="marimo logo">
</p>

<p align="center">
  <em>marimo: A reactive Python notebook that's reproducible, git-friendly, and deployable as scripts or apps.</em>
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
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads">
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" >
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

## marimo: The Reactive Python Notebook for Data Science

marimo is a revolutionary Python notebook that transforms how you work with data by offering a **reactive, reproducible, and shareable** environment.  With marimo, you can create interactive data science notebooks that are easy to version control, share as web apps, and execute as scripts.  [Learn more about marimo on GitHub](https://github.com/marimo-team/marimo).

**Key Features:**

*   🚀 **Batteries-Included:** Replaces `jupyter`, `streamlit`, `jupytext`, `ipywidgets`, `papermill`, and more.
*   ⚡️ **Reactive Execution:**  Changes in one cell automatically update dependent cells, ensuring data consistency.
*   🖐️ **Interactive UI Elements:**  Bind sliders, tables, plots, and other UI elements directly to your Python code with no callbacks.
*   🐍 **Git-Friendly:** Notebooks are stored as pure `.py` files, ideal for version control.
*   🛢️ **Data-Centric:** Built-in support for querying and manipulating data from dataframes, databases, and SQL warehouses.
*   🤖 **AI-Powered:** Generate cells with AI to accelerate your data workflow.
*   🔬 **Reproducible:**  Guaranteed execution order and no hidden state.
*   🏃 **Executable as Scripts:** Run notebooks as standard Python scripts with CLI argument support.
*   🛜 **Shareable:** Deploy notebooks as interactive web apps, slideshows, or run them in the browser via WASM.
*   🧩 **Reusable Code:** Import functions and classes from other notebooks for modularity.
*   🧪 **Testable:** Integrate pytest for comprehensive notebook testing.
*   ⌨️ **Modern Editor:** Features like GitHub Copilot integration, AI assistants, Vim keybindings, and a variable explorer enhance your coding experience.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [quickstart](#quickstart) for a primer on our CLI._

## How marimo Works: A Reactive Programming Environment

marimo ensures your code, outputs, and program state are consistent and predictable.  It addresses many of the common issues found in traditional notebooks.

**Reactive Programming:** marimo automatically re-runs cells that depend on variables changed in other cells, keeping your code and results synchronized. Delete a cell, and marimo scrubs its variables from program memory, eliminating hidden state and ensuring your notebook is always up-to-date.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Example">

**Handling "Expensive" Notebooks:**  marimo allows you to configure the runtime for lazy execution, marking affected cells as stale rather than re-running them automatically, preventing unnecessary computations.

**Synchronized UI Elements:** Easily create interactive notebooks with sliders, dropdowns, dataframe transformers, chat interfaces, and more, where the values are automatically updated.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI Elements Example">

**Interactive Dataframes:** Efficiently page through, filter, search, and sort millions of rows directly within your notebook.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Interactive Dataframes Example">

**AI-Assisted Code Generation:** Utilize an AI assistant to generate cells optimized for data work.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI Code Generation Example">

**SQL Integration:**  Build and execute SQL queries directly within your notebook, using dataframes, databases, lakehouses, CSV files, and more.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Integration Example">

**Dynamic Markdown:** Create dynamic markdown content that adapts to your Python data, telling engaging and insightful stories.

**Package Management:** Easily install packages on import with built-in support for major package managers.

**Deterministic Execution:**  Notebooks are executed in a deterministic order based on variable dependencies, allowing you to structure your notebooks for clarity.

**Optimized Performance:** marimo's runtime analyzes code dependencies to ensure efficient execution.

**Comprehensive Features:** Includes GitHub Copilot, AI assistants, Ruff code formatting, HTML export, a VS Code extension, and an interactive dataframe viewer.

## Quickstart

*The [marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq) on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an overview of many features.*

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

**Run Apps:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Run Apps example">

**Execute as Scripts:**

```bash
python your_notebook.py
```

**Convert Jupyter Notebooks:**

```bash
marimo convert your_notebook.ipynb > your_notebook.py
```
Or use our [web interface](https://marimo.io/convert).

**Tutorials:**

```bash
marimo tutorial --help
```

## FAQs

Visit the [FAQ](https://docs.marimo.io/faq.html) on our docs for answers to common questions.

## Learn More

marimo is designed for both beginners and experienced users.

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive Example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="UI Example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Plotting Example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Layout Example"/>
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
        <img src="https://marimo.io/shield.svg" alt="Try marimo online"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="marimo example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="marimo example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="marimo example"/>
      </a>
    </td>
  </tr>
</table>

Explore our [docs](https://docs.marimo.io), [usage examples](https://docs.marimo.io/examples/), and [gallery](https://marimo.io/gallery) to discover more.

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Questions?  Join us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join the marimo community!

*   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.** marimo is part of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFocus">

## Inspiration ✨

marimo reimagines the Python notebook for a reproducible, interactive, and shareable data science experience.

We believe in better tools for better thinking. marimo provides the Python community with a more powerful environment for research, communication, code experimentation, and education.

Our inspiration comes from projects like [Pluto.jl](https://github.com/fonsp/Pluto.jl), [ObservableHQ](https://observablehq.com/tutorials), and [Bret Victor's essays](http://worrydream.com/). marimo is part of a movement toward reactive dataflow programming.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo logo">
</p>
```

Key improvements and SEO considerations:

*   **Clear Heading Structure:** Uses `##` for all sub-sections, enhancing readability and SEO.
*   **Concise Hook:**  Starts with a strong, clear statement about what marimo *is* and its core benefits.
*   **Keyword Optimization:** Includes relevant keywords throughout (e.g., "reactive Python notebook," "data science," "interactive," "reproducible," "shareable," "Python scripts").
*   **Bulleted Key Features:**  Uses bullet points to highlight the core value propositions.
*   **Detailed Feature Descriptions:**  Provides more informative descriptions for each feature, including links to relevant documentation.
*   **Actionable Quickstart:**  Provides clear, easy-to-follow installation and usage instructions.
*   **Internal Linking:** Uses links to sections within the README (e.g., "[quickstart](#quickstart)") for improved navigation.
*   **Alt Text for Images:** Adds `alt` text to all images, crucial for accessibility and SEO.
*   **Community Section:**  Promotes community engagement with clear calls to action.
*   **Inspiration:** Acknowledges the core vision and inspiration behind the project.
*   **Links to Original Repo:** Maintained the link back to the original repo.
*   **Web Interface Link**: Added a link to the web interface for conversion.