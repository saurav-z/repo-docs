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

# marimo: The Reactive Python Notebook for Data Science

**marimo is a revolutionary Python notebook designed for modern data science, enabling reproducibility, interactivity, and seamless deployment.**  [Visit the marimo GitHub Repository](https://github.com/marimo-team/marimo)

**Key Features:**

*   🚀 **Batteries-Included:** Replaces `Jupyter`, `Streamlit`, `Jupytext`, `ipywidgets`, `papermill`, and more, offering a complete data science environment.
*   ⚡️ **Reactive Programming:** Automatically updates dependent cells when a cell's input changes, ensuring data consistency and simplifying debugging.
*   🖐️ **Interactive UIs:** Easily integrate interactive elements like sliders, tables, plots, and more into your notebooks without the need for callbacks.
*   🐍 **Git-Friendly:** Stores notebooks as standard `.py` files, making version control and collaboration a breeze.
*   🛢️ **Data-Centric:** Built-in support for querying and manipulating dataframes, databases, and more with SQL.
*   🤖 **AI-Powered:** Generate Python code with AI tailored for data work, streamlining your workflow and enhancing productivity.
*   🔬 **Reproducible:** No hidden state, deterministic execution order, and built-in package management ensure consistent results.
*   🏃 **Executable Scripts:** Execute notebooks as Python scripts with command-line arguments, offering flexibility in deployment.
*   🛜 **Shareable & Deployable:** Deploy your notebooks as interactive web apps, slides, or run them in the browser via WASM.
*   🧩 **Reusable:** Import and reuse functions and classes between notebooks, promoting code modularity and efficiency.
*   🧪 **Testable:** Easily integrate pytest for testing your notebooks, ensuring code quality and reliability.
*   ⌨️ **Modern Editor:** Enhanced with features like GitHub Copilot integration, AI assistants, Vim keybindings, and a variable explorer, offering a superior coding experience.

```python
pip install marimo && marimo tutorial intro
```

_Explore marimo instantly in our [online playground](https://marimo.app/l/c7h6pz), which runs directly in your browser!_

_Get started quickly with the [Quickstart](#quickstart) section._

## Core Principles & Benefits

marimo transforms the traditional notebook experience by ensuring consistency and reproducibility through reactive programming principles.  This means your code, outputs, and program state stay in sync.

**Reactive Environment:** Run a cell, and marimo *reacts* by automatically executing cells that rely on its outputs, eliminating manual re-running and potential errors.  Deleting a cell clears its variables from memory, preventing hidden state issues.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive GIF">

**Compatibility with Large Notebooks:**  Configure marimo's runtime for lazy evaluation, marking affected cells as "stale" instead of immediately running them. This preserves data integrity while avoiding the accidental execution of computationally expensive cells.

**Interactive UI Elements:** Seamlessly connect UI components like sliders, dropdowns, and dataframes to your Python code, with changes instantly reflected in dependent cells.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI Example GIF">

**Interactive DataFrames:** Effortlessly page, filter, sort, and search through datasets with millions of rows directly within your notebook.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Dataframe Example GIF">

**AI-Assisted Code Generation:** Leverage AI to generate code tailored for data tasks, with context from your current variables.  Even generate entire notebooks using a text description.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI Code Generation GIF">

**SQL Integration:**  Write and execute SQL queries that dynamically incorporate Python variables, retrieving results as Python dataframes from dataframes, databases, and more.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Example Image">

**Dynamic Markdown:**  Create dynamic markdown content that adapts to your Python variable values.

**Package Management:**  Easily manage Python packages using built-in support for major package managers, allowing installations on import, serialization of requirements within notebooks, and isolated environments.

**Deterministic Execution:** Notebooks are executed in a predictable order based on variable dependencies, rather than cell positions.

**High Performance:** marimo optimizes execution by running only the necessary cells through intelligent code analysis.

**Comprehensive Tooling:** Enjoy integrated features such as GitHub Copilot, AI assistants, Ruff code formatting, HTML export, fast code completion, a VS Code extension, an interactive dataframe viewer, and many more quality-of-life improvements.

## Quickstart

_Explore the core concepts of marimo by watching the [marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq) on our [YouTube channel](https://www.youtube.com/@marimo-team)._

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

For full features including SQL support and AI completion:

```bash
pip install marimo[recommended]
```

**Create and Edit Notebooks:**

```bash
marimo edit
```

**Run Notebooks as Web Apps:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Running as App GIF"/>

**Execute Notebooks as Scripts:**

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

## Support & Resources

Find answers to common questions in the [FAQ](https://docs.marimo.io/faq.html) section of our documentation.

## Learn More

marimo provides a powerful yet accessible environment, suitable for both beginners and experienced users.  Explore interactive examples like this embedding visualizer ([video](https://marimo.io/videos/landing/full.mp4)):

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/embedding.gif" width="700px" alt="Embedding Visualizer GIF"/>

Access our comprehensive [docs](https://docs.marimo.io), explore [usage examples](https://docs.marimo.io/examples/), and browse our [gallery](https://marimo.io/gallery) to delve deeper into marimo's features and capabilities.

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

We encourage and appreciate all contributions! See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details on how to get involved.

> Have questions? Join us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join our vibrant community and stay connected!

-   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
-   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
-   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
-   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
-   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
-   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
-   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
-   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
-   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.** marimo is part of the NumFOCUS community, supporting projects like NumPy, SciPy, and Matplotlib.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS"/>

## Our Inspiration

marimo reimagines the Python notebook as a reproducible, interactive, and shareable Python program, moving away from error-prone JSON structures.

We strive to create better tools for better minds. marimo aims to provide the Python community with a superior programming environment for research, communication, experimentation, and education in computational science.

We are inspired by projects such as [Pluto.jl](https://github.com/fonsp/Pluto.jl),
[ObservableHQ](https://observablehq.com/tutorials), and
[Bret Victor's essays](http://worrydream.com/), alongside the broader movement towards reactive dataflow programming, including [IPyflow](https://github.com/ipyflow/ipyflow), [streamlit](https://github.com/streamlit/streamlit),
[TensorFlow](https://github.com/tensorflow/tensorflow),
[PyTorch](https://github.com/pytorch/pytorch/tree/main),
[JAX](https://github.com/google/jax), and
[React](https://github.com/facebook/react).

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo logo horizontal">
</p>