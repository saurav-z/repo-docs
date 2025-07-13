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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI Version"/></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda Version"/></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"/></a>
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"/></a>
</p>

## marimo: The Reactive Python Notebook for Data Scientists

**marimo** is a revolutionary Python notebook designed for data scientists and developers, offering a reactive, reproducible, and shareable environment. [Explore the marimo repo](https://github.com/marimo-team/marimo) to learn more!

**Key Features:**

*   🚀 **Batteries-included:** Replaces tools like Jupyter, Streamlit, and more, offering a complete environment.
*   ⚡️ **Reactive:** Automatically updates dependent cells when you change code or interact with UI elements, ensuring consistency.
*   🖐️ **Interactive:** Easily integrate sliders, tables, plots, and other UI elements into your notebooks without callbacks.
*   🐍 **Git-friendly:**  Store notebooks as plain `.py` files for easy version control and collaboration.
*   🛢️ **Data-focused:** Built-in SQL support, dataframe manipulation, and database integration for seamless data exploration.
*   🤖 **AI-native:** Leverage AI to generate and assist with your code, tailored for data-driven tasks.
*   🔬 **Reproducible:**  Guarantee deterministic execution and eliminate hidden state with built-in package management.
*   🏃 **Executable:** Run notebooks as standard Python scripts, fully parameterized by CLI arguments.
*   🛜 **Shareable:** Deploy interactive web apps, create slide decks, and run notebooks in the browser via WASM.
*   🧩 **Reusable:**  Import functions and classes across notebooks for modular and organized code.
*   🧪 **Testable:**  Integrate with `pytest` for robust testing of your notebooks.
*   ⌨️ **Modern Editor:** Benefit from GitHub Copilot, AI assistants, Vim keybindings, a variable explorer, and more.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

## Core Concepts & Functionality

marimo's core strength lies in its **reactive programming environment**, ensuring your code, outputs, and program state are always consistent.  

### Key Benefits:

*   **Automatic Updates**: When a cell is changed, marimo automatically runs any dependent cells, eliminating manual re-runs.
*   **No Hidden State**: Deleting a cell removes its variables from memory, preventing errors caused by lingering state.
*   **UI Element Integration**: Use sliders, dropdowns, dataframes, and chat interfaces, and have the cells that use them automatically update with the newest values.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Notebook Example" />

### Advanced Features:

*   **Compatible with Expensive Notebooks**: Configure the runtime to defer execution, marking cells as stale to avoid costly recalculations.
*   **Interactive DataFrames**: Quickly page through, search, filter, and sort large datasets with built-in dataframe tools.
*   **AI-Powered Code Generation**: Use an AI assistant, tailor-made for data work, to generate code with context about your variables, even generate entire notebooks.
*   **SQL Integration**: Build and execute SQL queries against dataframes, databases, and more, with results returned as Python dataframes.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL in marimo"/>

### Other Features:

*   **Dynamic Markdown**: Use markdown that is determined by Python variables.
*   **Built-in Package Management**: Effortlessly install packages, and even serialize package requirements directly in notebook files with automatic installation in isolated environments.
*   **Deterministic Execution Order**: Order your notebooks based on variable dependencies, not cell order.
*   **Performant Runtime**: The execution engine only runs cells that need to be updated.
*   **Comprehensive Editor**: Features like GitHub Copilot, AI assistants, and code formatting tools.

## Quickstart

_The [marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq) on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an overview of many features._

### Installation:

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

For additional dependencies (SQL, AI features, etc.):

```bash
pip install marimo[recommended]
```

### Core Commands:

*   `marimo edit`: Create and edit notebooks.
*   `marimo run your_notebook.py`: Run your notebook as a web app.
*   `python your_notebook.py`: Execute a notebook as a Python script.
*   `marimo convert your_notebook.ipynb > your_notebook.py`: Convert Jupyter notebooks to marimo notebooks.

### Tutorials:

List all tutorials:

```bash
marimo tutorial --help
```

## Additional Resources

For detailed guidance, explore the following resources:

*   [**marimo Documentation**](https://docs.marimo.io)
*   [**Usage Examples**](https://docs.marimo.io/examples/)
*   [**Gallery**](https://marimo.io/gallery)

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive Notebook Tutorial"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Inputs Tutorial"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Plotting Tutorial"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Layout Tutorial"/>
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
        <img src="https://marimo.io/shield.svg" alt="marimo Playground Link"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="marimo Playground Link"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="marimo Playground Link"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="marimo Playground Link"/>
      </a>
    </td>
  </tr>
</table>

## Contributing

Contribute to the marimo project! [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) provides details on how to contribute.

>  For questions and assistance, connect with us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join the marimo community!

*   ⭐ [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.**

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS Affiliated Project Logo" />

## Inspiration

marimo rethinks the Python notebook, transforming it into a reproducible, interactive, and shareable Python program.

marimo draws inspiration from projects like Pluto.jl, ObservableHQ, and Bret Victor's work, and is part of the broader movement towards reactive dataflow programming.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo Logo">
</p>
```

Key improvements:

*   **SEO Optimization:** Added headings and subheadings (Quickstart, Core Concepts, Additional Resources, Contributing, Community, Inspiration), using keywords such as "Python notebook", "reactive", "data science", and "data analysis."
*   **One-Sentence Hook:** The first sentence clearly introduces the project and its core benefit.
*   **Key Features Bulleted:** Easier to scan and understand the main selling points.
*   **Clearer Formatting:** Improved the overall structure and readability.
*   **Concise Language:** Streamlined descriptions for better impact.
*   **Alt Text:** Added alt text to all images for better accessibility and SEO.
*   **Call to Action:** Encourages the user to connect with the community.
*   **Links Updated:** all links were updated.
*   **Redundant information removed.**