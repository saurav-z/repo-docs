<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg">
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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi"/></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg"/></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="discord" /></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" /></a>
</p>

## marimo: The Reactive Python Notebook for Data Science

**marimo is a cutting-edge Python notebook designed for modern data science workflows, offering reactivity, reproducibility, and seamless deployment.**

**Key Features:**

*   🚀 **Batteries Included:** Replaces multiple tools like Jupyter, Streamlit, and more, streamlining your workflow.
*   ⚡️ **Reactive Notebooks:** Experience automatic updates; changes in one cell trigger updates in dependent cells.
*   🖐️ **Interactive:** Easily bind UI elements like sliders, tables, and plots to your Python code for dynamic exploration.
*   🐍 **Git-Friendly:** Notebooks are stored as `.py` files, making version control and collaboration simple.
*   🛢️ **Data-Focused:** Query dataframes, databases, and more using SQL, plus explore and filter data directly.
*   🤖 **AI-Native:** Leverage AI to generate code tailored for data analysis.
*   🔬 **Reproducible:** Benefit from a deterministic execution order and built-in package management.
*   🏃 **Executable:** Run notebooks as Python scripts with command-line argument support.
*   🛜 **Shareable:** Deploy notebooks as interactive web apps or slides, or run them in the browser.
*   🧩 **Reusable:** Import functions and classes between notebooks for modularity.
*   🧪 **Testable:** Integrate pytest for robust testing of your notebooks.
*   ⌨️ **Modern Editor:** Enhanced with features like GitHub Copilot, AI assistants, and a modern editor experience.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [quickstart](#quickstart) for a primer on our CLI._

## What is marimo?

marimo is a reactive programming environment that ensures your notebook code, outputs, and program state remain consistent. This is achieved through a reactive architecture where cell execution is triggered by dependencies, eliminating the common problems of hidden state and manual cell re-running found in traditional notebooks.

**A reactive programming environment.** Run a cell and marimo _reacts_ by automatically running the cells that
reference its variables, eliminating the error-prone task of manually
re-running cells. Delete a cell and marimo scrubs its variables from program
memory, eliminating hidden state.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" />

**Compatible with expensive notebooks.** marimo lets you [configure the runtime
to be
lazy](https://docs.marimo.io/guides/configuration/runtime_configuration.html),
marking affected cells as stale instead of automatically running them. This
gives you guarantees on program state while preventing accidental execution of
expensive cells.

**Synchronized UI elements.** Interact with [UI
elements](https://docs.marimo.io/guides/interactivity.html) like [sliders](https://docs.marimo.io/api/inputs/slider.html#slider),
[dropdowns](https://docs.marimo.io/api/inputs/dropdown.html), [dataframe
transformers](https://docs.marimo.io/api/inputs/dataframe.html), and [chat
interfaces](https://docs.marimo.io/api/inputs/chat.html), and the cells that
use them are automatically re-run with their latest values.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" />

**Interactive dataframes.** [Page through, search, filter, and
sort](https://docs.marimo.io/guides/working_with_data/dataframes.html)
millions of rows blazingly fast, no code required.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" />

**Generate cells with data-aware AI.** [Generate code with an AI
assistant](https://docs.marimo.io/guides/editor_features/ai_completion/) that is highly
specialized for working with data, with context about your variables in memory;
[zero-shot entire notebooks](https://docs.marimo.io/guides/generate_with_ai/text_to_notebook/).
Customize the system prompt, bring your own API keys, or use local models.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" />

**Query data with SQL.** Build [SQL](https://docs.marimo.io/guides/working_with_data/sql.html) queries
that depend on Python values and execute them against dataframes, databases, lakehouses,
CSVs, Google Sheets, or anything else using our built-in SQL engine, which
returns the result as a Python dataframe.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" />

Your notebooks are still pure Python, even if they use SQL.

**Dynamic markdown.** Use markdown parametrized by Python variables to tell
dynamic stories that depend on Python data.

**Built-in package management.** marimo has built-in support for all major
package managers, letting you [install packages on import](https://docs.marimo.io/guides/editor_features/package_management.html). marimo can even
[serialize package
requirements](https://docs.marimo.io/guides/package_management/inlining_dependencies/)
in notebook files, and auto install them in
isolated venv sandboxes.

**Deterministic execution order.** Notebooks are executed in a deterministic
order, based on variable references instead of cells' positions on the page.
Organize your notebooks to best fit the stories you'd like to tell.

**Performant runtime.** marimo runs only those cells that need to be run by
statically analyzing your code.

**Batteries-included.** marimo comes with GitHub Copilot, AI assistants, Ruff
code formatting, HTML export, fast code completion, a [VS Code
extension](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo),
an interactive dataframe viewer, and [many more](https://docs.marimo.io/guides/editor_features/index.html)
quality-of-life features.

## Quickstart

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

**Create & Edit:**

```bash
marimo edit
```

**Run as Web App:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" />

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

**Share Cloud-Based Notebooks:** Use [molab](https://molab.marimo.io/notebooks) for cloud-based sharing.

## Questions?

See the [FAQ](https://docs.marimo.io/faq.html) in our docs.

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

See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Questions? Reach out to us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join the marimo community!

- 🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
- 💬 [Chat on Discord](https://marimo.io/discord?ref=readme)
- 📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
- ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
- ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
- 🦋 [Follow on Bluesky](https://bsky.app/profile/marimo.io)
- 🐦 [Follow on Twitter](https://twitter.com/marimo_io)
- 🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
- 🕴️ [Follow on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.** [marimo](https://github.com/marimo-team/marimo) is part of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" />

## Inspiration ✨

marimo is a reinvention of the Python notebook, focusing on reproducibility, interaction, and shareability.

We are inspired by projects like [Pluto.jl](https://github.com/fonsp/Pluto.jl), [ObservableHQ](https://observablehq.com/tutorials), and the ideas of [Bret Victor](http://worrydream.com/).

[Go to the marimo GitHub repository](https://github.com/marimo-team/marimo)