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

**marimo is a modern Python notebook that transforms how you work with data, offering reactivity, reproducibility, and effortless deployment.**  [Get started with marimo](https://github.com/marimo-team/marimo) today!

**Key Features:**

*   🚀 **Batteries-Included:** Replaces Jupyter, Streamlit, and more, offering a comprehensive environment.
*   ⚡️ **Reactive:** Automatically re-runs dependent cells when a cell's input changes, ensuring consistency.
*   🖐️ **Interactive:** Easily bind UI elements (sliders, dropdowns, etc.) to your Python code without callbacks.
*   🐍 **Git-Friendly:** Stores notebooks as pure `.py` files for easy version control and collaboration.
*   🛢️ **Data-Focused:** Built-in SQL support for querying dataframes, databases, and data warehouses.
*   🤖 **AI-Native:** Generate code with AI assistance tailored for data work, including zero-shot notebook generation.
*   🔬 **Reproducible:**  No hidden state, deterministic execution, and built-in package management for reliable results.
*   🏃 **Executable:**  Run notebooks as standard Python scripts, parameterized by command-line arguments.
*   🛜 **Shareable:** Deploy your notebooks as interactive web apps or slides, and even run them in the browser using WASM.
*   🧩 **Reusable:** Import functions and classes between notebooks for modularity.
*   🧪 **Testable:**  Integrate with `pytest` to ensure your notebooks are reliable.
*   ⌨️ **Modern Editor:** Enhanced with GitHub Copilot, AI assistants, Vim keybindings, and a variable explorer.

```python
pip install marimo && marimo tutorial intro
```

*Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!*

## A Deeper Dive into marimo

marimo guarantees your notebook code, outputs, and program state are consistent, solving common problems associated with traditional notebooks like Jupyter.

**Reactive Programming Environment:**
When you run a cell, marimo reacts by automatically running cells that reference its variables, which eliminates the error-prone task of manually re-running cells.

**Compatible with Expensive Notebooks:** With marimo, configure the runtime to be lazy, marking affected cells as stale instead of automatically running them. This gives you guarantees on program state while preventing accidental execution of expensive cells.

**Synchronized UI Elements:** Interact with UI elements, such as sliders, dropdowns, dataframe transformers, and chat interfaces, and the cells that use them are automatically re-run with their latest values.

**Interactive Dataframes:** Page through, search, filter, and sort millions of rows blazingly fast, with no code required.

**Generate Cells with Data-Aware AI:** Generate code with an AI assistant that is highly specialized for working with data, with context about your variables in memory; zero-shot entire notebooks. Customize the system prompt, bring your own API keys, or use local models.

**Query Data with SQL:** Build SQL queries that depend on Python values and execute them against dataframes, databases, lakehouses, CSVs, Google Sheets, or anything else using our built-in SQL engine, which returns the result as a Python dataframe.

**Dynamic Markdown:** Use markdown parametrized by Python variables to tell dynamic stories that depend on Python data.

**Built-in Package Management:** marimo has built-in support for all major package managers, letting you install packages on import. marimo can even serialize package requirements in notebook files, and auto install them in isolated venv sandboxes.

**Deterministic Execution Order:** Notebooks are executed in a deterministic order, based on variable references instead of cells' positions on the page. Organize your notebooks to best fit the stories you'd like to tell.

**Performant Runtime:** marimo runs only those cells that need to be run by statically analyzing your code.

## Quickstart

_The [marimo concepts
playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq)
on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an
overview of many features._

**Installation:** In a terminal, run:

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with additional dependencies that unlock SQL cells, AI completion, and more, run:

```bash
pip install marimo[recommended]
```

**Create notebooks:**

Create or edit notebooks with

```bash
marimo edit
```

**Run apps:** Run your notebook as a web app, with Python code hidden and uneditable:

```bash
marimo run your_notebook.py
```

**Execute as scripts:** Execute a notebook as a script at the command line:

```bash
python your_notebook.py
```

**Automatically convert Jupyter notebooks:** Automatically convert Jupyter notebooks to marimo notebooks with the CLI

```bash
marimo convert your_notebook.ipynb > your_notebook.py
```

or use our [web interface](https://marimo.io/convert).

**Tutorials:**
List all tutorials:

```bash
marimo tutorial --help
```

**Share cloud-based notebooks:** Use
[molab](https://molab.marimo.io/notebooks), a cloud-based marimo notebook
service similar to Google Colab, to create and share notebook links.

## Questions?

See the [FAQ](https://docs.marimo.io/faq.html) at our docs.

## Learn More

marimo is easy to get started with, with lots of room for power users.
For example, here's an embedding visualizer made in marimo
([video](https://marimo.io/videos/landing/full.mp4)):

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/embedding.gif" width="700px" />

Check out our [docs](https://docs.marimo.io),
[usage examples](https://docs.marimo.io/examples/), and our [gallery](https://marimo.io/gallery) to learn more.

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

We appreciate all contributions! You don't need to be an expert to help out.
Please see [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for more details on how to get
started.

> Questions? Reach out to us [on Discord](https://marimo.io/discord?ref=readme).

## Community

We're building a community. Come hang out with us!

- 🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
- 💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
- 📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
- ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
- ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
- 🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
- 🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
- 🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
- 🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.** marimo is a core part of the broader Python
ecosystem and is a member of the NumFOCUS community, which includes projects
such as NumPy, SciPy, and Matplotlib.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" />

## Inspiration ✨

marimo is a **reinvention** of the Python notebook as a reproducible, interactive,
and shareable Python program, instead of an error-prone JSON scratchpad.

We believe that the tools we use shape the way we think — better tools, for
better minds. With marimo, we hope to provide the Python community with a
better programming environment to do research and communicate it; to experiment
with code and share it; to learn computational science and teach it.

Our inspiration comes from many places and projects, especially
[Pluto.jl](https://github.com/fonsp/Pluto.jl),
[ObservableHQ](https://observablehq.com/tutorials), and
[Bret Victor's essays](http://worrydream.com/). marimo is part of
a greater movement toward reactive dataflow programming. From
[IPyflow](https://github.com/ipyflow/ipyflow), [streamlit](https://github.com/streamlit/streamlit),
[TensorFlow](https://github.com/tensorflow/tensorflow),
[PyTorch](https://github.com/pytorch/pytorch/tree/main),
[JAX](https://github.com/google/jax), and
[React](https://github.com/facebook/react), the ideas of functional,
declarative, and reactive programming are transforming a broad range of tools
for the better.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px">
</p>