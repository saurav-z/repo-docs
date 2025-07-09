<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="marimo logo">
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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI Version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda Version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"></a>
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads" />
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

## Marimo: The Reactive Python Notebook for Modern Data Science

**Marimo is a revolutionary Python notebook that reimagines how you code, analyze, and share your work, offering a streamlined, interactive, and reproducible environment.**  [Get started with marimo on GitHub](https://github.com/marimo-team/marimo).

**Key Features:**

*   🚀 **Batteries-included:** Replaces Jupyter, Streamlit, and more, providing a comprehensive environment.
*   ⚡️ **Reactive:** Automatically updates dependent cells when you run a cell or interact with a UI element.
*   🖐️ **Interactive:** Easily bind interactive elements like sliders and plots to your Python code, no callbacks needed.
*   🐍 **Git-friendly:** Notebooks are saved as `.py` files, making version control a breeze.
*   🛢️ **Designed for Data:**  Seamlessly work with data using SQL queries, dataframe manipulation, and more.
*   🤖 **AI-Native:** Generate code with AI assistants specifically designed for data work, tailored to your project.
*   🔬 **Reproducible:** Ensures consistent results with no hidden state and built-in package management.
*   🏃 **Executable:** Run your notebooks as Python scripts with CLI arguments.
*   🛜 **Shareable:** Deploy notebooks as interactive web apps or slides, and run them in the browser via WASM.
*   🧩 **Reusable:** Import functions and classes from other notebooks.
*   🧪 **Testable:** Integrate pytest to ensure notebook code quality.
*   ⌨️ **Modern Editor:** Includes a modern editor with GitHub Copilot, AI assistants, Vim keybindings, and a variable explorer.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [quickstart](#quickstart) for a primer on our CLI._

## Core Concepts

marimo transforms the traditional notebook experience, ensuring your code, outputs, and program state remain consistent and reproducible.

**Reactive Programming Environment:**  When you run a cell, marimo automatically executes any cells that depend on it, eliminating the need for manual re-runs. Deleting a cell removes its variables from memory, avoiding hidden state issues.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="marimo reactivity in action" />

**Efficient Handling of Complex Notebooks:**  Configure marimo to lazily update cells, preventing accidental execution of expensive computations.

<a name="expensive-notebooks"></a>

**Synchronized UI Elements:** Use UI elements such as sliders, dropdowns, and dataframes, and see your code update in real-time.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="marimo interactive UI elements"/>

**Interactive DataFrames:**  Quickly explore and analyze large datasets directly within the notebook environment.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="marimo interactive dataframes"/>

**AI-Powered Code Generation:**  Leverage an AI assistant for code generation that is specialized for data work, offering context about your variables and enabling zero-shot notebook creation.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="marimo AI code generation"/>

**SQL Integration:** Query data using SQL, which can depend on Python values. Execute against dataframes, databases, and more.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="marimo SQL integration"/>

**Dynamic Markdown & Built-in Package Management:** Create markdown that is parametrized by Python variables, and use built-in support for all major package managers.

**Deterministic Execution:** Notebooks execute in a predictable order based on variable dependencies, not cell order.

**High-Performance Runtime:**  marimo's runtime is designed for performance, executing only the necessary cells.

**Comprehensive Features:**  Benefit from features such as GitHub Copilot, AI assistants, fast code completion, a VS Code extension, and an interactive dataframe viewer.

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

**Create and Edit Notebooks:**

```bash
marimo edit
```

**Run as Web Apps:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="marimo app demonstration"/>

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

## Need Help?

Refer to our [FAQ](https://docs.marimo.io/faq.html) for answers to common questions.

## Learn More

marimo offers both ease of use for beginners and advanced features for experienced users. Explore the possibilities!

Here's an embedding visualizer made in marimo:
([video](https://marimo.io/videos/landing/full.mp4)):

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/embedding.gif" width="700px" alt="marimo embedding visualizer" />

Explore our [docs](https://docs.marimo.io),
[usage examples](https://docs.marimo.io/examples/), and our [gallery](https://marimo.io/gallery) to learn more.

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="marimo tutorial" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="marimo inputs" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="marimo plots" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="marimo layouts" />
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
        <img src="https://marimo.io/shield.svg" alt="marimo playground"/>
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

## Contributing

Your contributions are welcome! Refer to [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for guidance.

> Reach out to us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join our community!

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

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS affiliated project" />

## Inspiration ✨

marimo is a **reinvention** of the Python notebook, aiming to be a reproducible, interactive, and shareable Python program.

We aim to provide the Python community with a better programming environment.

Our inspiration includes [Pluto.jl](https://github.com/fonsp/Pluto.jl), [ObservableHQ](https://observablehq.com/tutorials), and [Bret Victor's essays](http://worrydream.com/). marimo is part of a greater movement toward reactive dataflow programming.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo logo">
</p>
```
Key improvements and SEO optimizations:

*   **Clear and Concise Title:** Added a strong, SEO-friendly title: "Marimo: The Reactive Python Notebook for Modern Data Science".
*   **One-Sentence Hook:** Provided a compelling one-sentence description at the beginning to grab the reader's attention.
*   **Keyword Optimization:** Incorporated relevant keywords like "reactive Python notebook," "data science," "interactive," "reproducible," "Git-friendly," and "web app" throughout the README.
*   **Header Hierarchy:** Structured the README with clear headings (##) to improve readability and SEO.
*   **Bulleted Key Features:**  Organized the key features in a bulleted list for easy scanning and understanding.
*   **Alt Text for Images:** Added `alt` text to all images for accessibility and SEO benefits.
*   **Descriptive Text:** Added descriptive text to all graphics with explanations and keywords.
*   **Call to Actions:** Incorporated calls to action, encouraging users to visit the docs, examples, and other resources.
*   **Expanded Quickstart:**  Added more context to the quickstart to make it easier for new users.
*   **Community Section:** Enhanced the community section to promote engagement.
*   **Emphasis on Benefits:** Clearly highlighted the benefits of using marimo, such as reproducibility and interactivity.
*   **Concise Language:** Simplified language where possible to improve clarity.
*   **Reorganized Content:** Streamlined the structure for a better user experience.
*   **Improved Readability:**  Formatted the markdown for better readability.