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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI Version"/></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda Version"/></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord" /></a>
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License" /></a>
</p>

## marimo: The Reactive Python Notebook for Data Science

**marimo** is a revolutionary, open-source Python notebook designed to streamline data science workflows, making your code reproducible, shareable, and easily deployed.  Build data-driven applications with a modern notebook experience.  Check out the [original repo](https://github.com/marimo-team/marimo) for more details.

**Key Features:**

*   🚀 **Batteries-Included:** A comprehensive solution, replacing tools like Jupyter, Streamlit, and more.
*   ⚡️ **Reactive Execution:**  Changes to a cell automatically update dependent cells, ensuring data consistency.
*   🖐️ **Interactive UI:** Seamlessly integrate interactive elements like sliders, plots, and tables directly into your notebooks without callbacks.
*   🐍 **Git-Friendly:** Notebooks are stored as standard `.py` files for easy version control.
*   🛢️ **Data-Centric:**  Built-in support for SQL queries, dataframe manipulation, and integration with databases and data warehouses.
*   🤖 **AI-Powered:** Enhance your workflow with AI-driven code generation, tailored for data science tasks.
*   🔬 **Reproducible Results:** Guarantees no hidden state, deterministic execution, and built-in package management.
*   🏃 **Executable Scripts:**  Easily run notebooks as Python scripts, configurable with command-line arguments.
*   🛜 **Shareable & Deployable:**  Deploy notebooks as interactive web apps or presentations, even run them in the browser using WASM.
*   🧩 **Code Reusability:** Import functions and classes across notebooks for efficient code reuse.
*   🧪 **Testable Notebooks:** Leverage pytest for comprehensive testing of your notebooks.
*   ⌨️ **Modern Editor:** Benefit from a modern code editor with features like GitHub Copilot integration, AI assistants, Vim keybindings, and a variable explorer.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [quickstart](#quickstart) for a primer on our CLI._

## How marimo Works: A Reactive Programming Environment

marimo introduces a new way to interact with your data science projects.  Instead of the manual cell execution found in traditional notebooks, marimo uses a reactive system to provide a more predictable and efficient workflow.

**Key Benefits:**

*   **Automatic Updates:** Run a cell, and marimo intelligently re-executes any dependent cells, ensuring your outputs always reflect the latest data.
*   **Eliminate Hidden State:** Delete a cell, and marimo automatically removes its variables from program memory, improving reproducibility.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="marimo reactivity GIF" />

<a name="expensive-notebooks"></a>

**Optimized for Performance:**  For notebooks with expensive operations, marimo allows you to configure a "lazy" runtime.  This marks dependent cells as stale instead of automatically re-running them, preventing accidental execution of time-consuming code.

**Interactive Data Exploration:**

*   **UI Elements:**  Use sliders, dropdowns, dataframe transformers, and chat interfaces to create interactive notebooks.
*   **Interactive Dataframes:** Page through, filter, and sort millions of rows of data directly within your notebook.
*   **AI-Assisted Code Generation:** Generate Python code with AI assistants for SQL queries and data exploration.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="marimo UI GIF" />

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="marimo dataframe GIF" />

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="marimo AI assist GIF" />

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="marimo SQL cell image" />

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

Install additional dependencies to unlock features like SQL cells and AI:

```bash
pip install marimo[recommended]
```

**Getting Started:**

*   **Create/Edit Notebooks:** `marimo edit`
*   **Run Apps:** `marimo run your_notebook.py`
*   **Run as Script:** `python your_notebook.py`
*   **Convert Jupyter Notebooks:** `marimo convert your_notebook.ipynb > your_notebook.py` or use our [web interface](https://marimo.io/convert).
*   **Explore Tutorials:** `marimo tutorial --help`

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Marimo App Run Example"/>

## Questions & Support

For answers to common questions, check out the [FAQ](https://docs.marimo.io/faq.html) in the documentation.

## Learn More

marimo offers powerful features for both beginners and advanced users. See the interactive embedding visualizer, powered by marimo:

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/embedding.gif" width="700px" alt="marimo embedding visualizer GIF" />

Explore the [docs](https://docs.marimo.io), [examples](https://docs.marimo.io/examples/), and [gallery](https://marimo.io/gallery) to uncover the full potential of marimo.

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
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Layouts"/>
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
        <img src="https://marimo.io/shield.svg" alt="Marimo Demo"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Marimo Demo"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Marimo Demo"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Marimo Demo"/>
      </a>
    </td>
  </tr>
</table>

## Contributing

We welcome contributions!  Review [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details on how to contribute.

> Need Help?  Join the conversation [on Discord](https://marimo.io/discord?ref=readme).

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

**A NumFOCUS Affiliated Project.** marimo is part of the NumFOCUS community.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS Logo" />

## Inspiration ✨

marimo is a re-imagining of the Python notebook, built to be a reliable, interactive, and shareable Python program instead of a potentially error-prone scratchpad. Our goal is to provide the Python community with tools that enable better research, collaboration, and communication, and that help you do more with your data.

We draw inspiration from projects like [Pluto.jl](https://github.com/fonsp/Pluto.jl) and [ObservableHQ](https://observablehq.com/tutorials), and from the ideas of [Bret Victor](http://worrydream.com/). marimo contributes to the growing movement toward reactive dataflow programming.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo logo">
</p>
```
Key improvements and explanations:

*   **SEO Optimization:**  Incorporated relevant keywords like "Python notebook," "reactive," "data science," "interactive," "reproducible," "Git-friendly," and "deployable" throughout the README, especially in the headings and the introduction.
*   **Concise Hook:**  The one-sentence hook immediately grabs the reader's attention.
*   **Clear Headings:**  Used clear and descriptive headings to structure the information logically.
*   **Bulleted Key Features:**  Organized the main features into an easy-to-scan bulleted list, making it easy for users to quickly understand the capabilities of marimo.
*   **Strong Emphasis on Benefits:**  Focuses on the *benefits* of using marimo, not just the features (e.g., "Git-Friendly" is followed by "for easy version control").
*   **Improved Descriptions:** Expanded on the descriptions of the key features for better understanding.
*   **Visual Cues:** Added `alt` text to the images and clarified the function of the animated GIFs to help search engines and users.
*   **Call to Action:** The "Quickstart" section provides clear instructions on how to install and get started.
*   **Community & Contact Information:**  Made it easy for users to connect with the project team and other users.
*   **Inspiration Section:** Added the "Inspiration" section to explain the project's vision and its place in the broader movement.
*   **Alt Text for Images:**  Added descriptive `alt` text to all images to improve accessibility and SEO.
*   **Removed Unnecessary Details:** Streamlined some of the less important text to improve readability.
*   **Corrected Badges:** Fixed the broken badges to ensure they render correctly, and added descriptions to provide context.