<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="marimo logo">
</p>

<p align="center">
  <em>Unlock the power of reactive Python notebooks with marimo, transforming your code into reproducible, interactive, and shareable experiences.</em>
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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"></a>
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

## What is marimo?

marimo is a revolutionary, **reactive Python notebook** that allows you to create reproducible, git-friendly, and deployable data science projects. Build interactive apps, share dynamic presentations, and streamline your workflow with a modern take on the notebook experience.

## Key Features

*   🚀 **Batteries-included:** Replaces tools like Jupyter, Streamlit, and more, offering a comprehensive environment.
*   ⚡️ **Reactive:** Automatically updates dependent cells when a cell's output changes, or marks them as stale.  Learn more about [marimo's reactivity](https://docs.marimo.io/guides/reactivity.html).
*   🖐️ **Interactive:** Easily create interactive elements like sliders, tables, and plots, directly linked to your Python code, with no callbacks required.  See [marimo's interactivity guide](https://docs.marimo.io/guides/interactivity.html).
*   🐍 **Git-friendly:** Store notebooks as `.py` files for easy version control and collaboration.
*   🛢️ **Data-centric:**  Work with data seamlessly, including SQL queries, dataframe transformations, and database integration. Use [marimo's SQL support](https://docs.marimo.io/guides/working_with_data/sql.html).
*   🤖 **AI-native:** Generate and refine code with AI assistance, customized for data work.  Get started with [AI-powered code generation](https://docs.marimo.io/guides/generate_with_ai/).
*   🔬 **Reproducible:**  Ensure consistent results with deterministic execution and built-in package management.  Explore [package management in marimo](https://docs.marimo.io/guides/package_management/).
*   🏃 **Executable:** Run your notebooks as Python scripts with command-line argument support. See [how to execute notebooks as scripts](https://docs.marimo.io/guides/scripts.html).
*   🛜 **Shareable:** Deploy notebooks as interactive web apps or presentations, and even run them in the browser. Learn more about [marimo apps](https://docs.marimo.io/guides/apps.html).
*   🧩 **Reusable:** Import functions and classes between notebooks for modular code.
*   🧪 **Testable:** Integrate with pytest for robust testing of your notebooks.
*   ⌨️ **Modern Editor:** Benefit from features like GitHub Copilot, AI assistants, Vim keybindings, and a variable explorer.  Explore [marimo's editor features](https://docs.marimo.io/guides/editor_features/index.html).

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

## A New Approach to Notebooks

marimo rethinks the traditional notebook paradigm, offering a reactive programming environment that ensures consistency and reproducibility. Solve the challenges of hidden state and manual cell execution with marimo's intelligent reactivity.

**Reactive Programming:**  Run a cell and marimo intelligently updates dependent cells.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive animation">

**Optimized for Complex Workflows:** Configure the runtime to be lazy and mark affected cells as stale to prevent accidental execution of expensive cells.  See the [runtime configuration guide](https://docs.marimo.io/guides/configuration/runtime_configuration.html).

**Interactive User Interfaces:** Integrate UI elements like sliders, dropdowns, and dataframes, and see the connected cells update in real-time.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI animation">

**Interactive Dataframes:** Easily page, filter, search, and sort through millions of rows directly within your notebook.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Dataframe animation">

**AI-Assisted Code Generation:** Leverage AI to generate and refine code, including entire notebooks.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI code generation animation">

**SQL Integration:** Use SQL queries directly within your notebooks to query data from dataframes, databases, and more.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL cell example">

**Dynamic Markdown:** Create dynamic and engaging narratives using markdown that is parametrized by Python variables.

**Integrated Package Management:**  Install and manage dependencies directly within your notebooks.

**Deterministic Execution:** Notebooks are executed in a deterministic order, based on variable references.

**Optimized Performance:** marimo executes only the necessary cells to maximize efficiency.

## Quickstart

Get started with marimo in minutes.

_The [marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq) on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an overview of many features._

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with additional dependencies that unlock SQL cells, AI completion, and more,
run:

```bash
pip install marimo[recommended]
```

**Create Notebooks:**

```bash
marimo edit
```

**Run as Web App:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Model comparison animation">

**Execute as Script:**

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

**Share Cloud-Based Notebooks:** Use [molab](https://molab.marimo.io/notebooks) for cloud-based collaboration.

## Get Help

*   Find answers in the [FAQ](https://docs.marimo.io/faq.html).

## Learn More

Discover the possibilities of marimo.

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive concepts">
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
        <img src="https://marimo.io/shield.svg" alt="Marimo Playground">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Marimo example">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Marimo example">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Marimo example">
      </a>
    </td>
  </tr>
</table>

## Contribute

We welcome contributions! See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

>  Need help?  Join us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join the marimo community:

*   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.** marimo is part of the NumFOCUS community, alongside projects like NumPy, SciPy, and Matplotlib.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS logo" />

## Inspiration

marimo is inspired by projects like Pluto.jl and ObservableHQ, and strives to be a better programming environment for research, communication, and education.

For more information about marimo, visit the [original repo](https://github.com/marimo-team/marimo).