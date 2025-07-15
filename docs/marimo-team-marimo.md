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


# marimo: The Reactive Python Notebook for Data Science

**marimo** is a next-generation Python notebook that offers a reactive programming environment, making your code reproducible, shareable, and ideal for data science projects.  Explore the [marimo](https://github.com/marimo-team/marimo) project on GitHub.

**Key Features:**

*   🚀 **Batteries-Included**: Replace tools like Jupyter, Streamlit, and more with marimo's comprehensive features.
*   ⚡️ **Reactive**: Automatically updates dependent cells when you change code or interact with UI elements.
*   🖐️ **Interactive**: Create interactive elements like sliders, tables, and plots without complex callback functions.
*   🐍 **Git-Friendly**: Store notebooks as `.py` files for seamless version control.
*   🛢️ **Designed for Data**: Easily query and manipulate data with SQL and built-in dataframe capabilities.
*   🤖 **AI-Native**: Generate code with AI assistants specifically designed for data work.
*   🔬 **Reproducible**: Ensures consistent results with no hidden state and built-in package management.
*   🏃 **Executable**: Run notebooks as Python scripts with command-line argument support.
*   🛜 **Shareable**: Deploy notebooks as interactive web apps, slides, or run them in the browser.
*   🧩 **Reusable**: Import functions and classes between notebooks for code reuse.
*   🧪 **Testable**: Easily integrate notebooks with pytest for thorough testing.
*   ⌨️ **Modern Editor**: Includes features like GitHub Copilot integration, AI assistants, Vim keybindings, and more.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

## Core Concepts

marimo ensures your notebook code, outputs, and program state are consistent, solving many problems associated with traditional notebooks.

**A Reactive Programming Environment** Run a cell and marimo _reacts_ by automatically running the cells that
reference its variables, eliminating the error-prone task of manually
re-running cells. Delete a cell and marimo scrubs its variables from program
memory, eliminating hidden state.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive example">

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

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI example">

**Interactive dataframes.** [Page through, search, filter, and
sort](https://docs.marimo.io/guides/working_with_data/dataframes.html)
millions of rows blazingly fast, no code required.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Dataframe example">

**Generate cells with data-aware AI.** [Generate code with an AI
assistant](https://docs.marimo.io/guides/editor_features/ai_completion/) that is highly
specialized for working with data, with context about your variables in memory;
[zero-shot entire notebooks](https://docs.marimo.io/guides/generate_with_ai/text_to_notebook/).
Customize the system prompt, bring your own API keys, or use local models.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI generate example">

**Query data with SQL.** Build [SQL](https://docs.marimo.io/guides/working_with_data/sql.html) queries
that depend on Python values and execute them against dataframes, databases, lakehouses,
CSVs, Google Sheets, or anything else using our built-in SQL engine, which
returns the result as a Python dataframe.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Example">

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

**Create & Edit Notebooks:**

```bash
marimo edit
```

**Run Apps:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Model Comparison">

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

## Get Started

Ready to revolutionize your data science workflow?  Explore the following resources:

*   [Documentation](https://docs.marimo.io)
*   [Usage Examples](https://docs.marimo.io/examples/)
*   [Gallery](https://marimo.io/gallery)

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive Tutorial">
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
        <img src="https://marimo.io/shield.svg" alt="Interactive playground example">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Interactive playground example">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Interactive playground example">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Interactive playground example">
      </a>
    </td>
  </tr>
</table>

## Contributing

Contributions are welcome!  Review the [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

>   Reach out [on Discord](https://marimo.io/discord?ref=readme) for questions.

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

**A NumFOCUS Affiliated Project:** marimo is part of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS logo">

## Inspiration

marimo reimagines Python notebooks for a more reproducible, interactive, and shareable experience.

We are inspired by [Pluto.jl](https://github.com/fonsp/Pluto.jl), [ObservableHQ](https://observablehq.com/tutorials), and [Bret Victor's essays](http://worrydream.com/). marimo is part of
a greater movement toward reactive dataflow programming.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo Logo Horizontal">
</p>
```

Key changes and improvements:

*   **SEO Optimization**: Added headings (H1, H2), targeted keywords (e.g., "reactive Python notebook," "data science," "interactive," "reproducible"), and used descriptive text.
*   **Summarization**: Concise descriptions, focusing on key benefits.
*   **Readability**: Improved formatting with consistent use of bolding and bullet points.
*   **Conciseness**: Removed redundant information.
*   **Call to Action**: Encouraged exploration with links to resources and community.
*   **Alt Text**: Added `alt` text to all images for accessibility and SEO.
*   **Removed unnecessary links**: Consolidated repeated links.
*   **Improved visuals**: Added descriptions to the images to further assist SEO.