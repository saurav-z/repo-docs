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

# marimo: The Reactive Python Notebook for Data Science

**marimo is a revolutionary Python notebook that transforms how you work with code, offering a reproducible, git-friendly, and deployable environment for data science and beyond.**  [Explore the project on GitHub](https://github.com/marimo-team/marimo).

**Key Features:**

*   🚀 **Batteries-included:** Replaces tools like Jupyter, Streamlit, and more, offering a unified experience.
*   ⚡️ **Reactive:** Automatically runs dependent cells when a cell is updated, ensuring consistency.
*   🖐️ **Interactive:** Easily bind UI elements like sliders and plots to Python code without callbacks.
*   🐍 **Git-friendly:**  Notebooks are stored as `.py` files, making version control seamless.
*   🛢️ **Designed for Data:**  Query and manipulate data using SQL, and filter dataframes.
*   🤖 **AI-native**:  Generate cells with AI assistance tailored for data work.
*   🔬 **Reproducible:** Ensures deterministic execution and eliminates hidden state.
*   🏃 **Executable:** Run notebooks as Python scripts with CLI argument support.
*   🛜 **Shareable:** Deploy notebooks as interactive web apps or slides; run in the browser via WASM.
*   🧩 **Reusable:** Import functions and classes between notebooks.
*   🧪 **Testable:** Integrate with pytest for robust testing.
*   ⌨️ **Modern Editor:** Benefit from features like GitHub Copilot integration, AI assistants, and more.

## Getting Started

### Installation

Install marimo using pip or conda:

```bash
pip install marimo  # or conda install -c conda-forge marimo
```

To install with recommended dependencies (SQL cells, AI completion):

```bash
pip install marimo[recommended]
```

### Basic Commands

*   **Create/Edit Notebooks:** `marimo edit`
*   **Run as Web App:** `marimo run your_notebook.py`
*   **Execute as Script:** `python your_notebook.py`
*   **Convert Jupyter Notebooks:** `marimo convert your_notebook.ipynb > your_notebook.py` or use the [web interface](https://marimo.io/convert).
*   **List Tutorials:** `marimo tutorial --help`

## A Deeper Dive into marimo's Capabilities

marimo provides a streamlined and efficient environment for data scientists, researchers, and developers. It addresses the limitations of traditional notebooks by incorporating reactivity and offering a suite of features for data exploration, analysis, and presentation.

**Key features:**

*   **Reactive Programming Environment:** marimo intelligently manages cell execution, automatically updating dependent cells to maintain code and output consistency, which dramatically reduces errors and improves workflow efficiency.
*   **Compatible with Expensive Notebooks:** You can configure marimo to mark computationally expensive cells as stale, preventing accidental execution while maintaining program state guarantees.
*   **Synchronized UI Elements:** Easily create interactive notebooks by binding UI elements (sliders, dropdowns, dataframes, chat interfaces) to your Python code, with automatic re-execution upon interaction.
*   **Interactive Dataframes:** Explore large datasets with built-in tools for filtering, sorting, and searching within the notebook environment.
*   **AI-Assisted Code Generation:**  Leverage AI assistants to generate code, tailored to data analysis tasks, providing context based on your variables in memory, and facilitating the creation of zero-shot notebooks.
*   **SQL Queries:** Integrate SQL queries seamlessly, allowing you to query data from dataframes, databases, and other sources.
*   **Dynamic Markdown:**  Create dynamic narratives with markdown, parametrized by Python variables.
*   **Built-in Package Management:** Simplifies package installation and dependency management with support for all major package managers.
*   **Deterministic Execution:** Ensures that notebooks execute in a predictable order, making organization intuitive.
*   **High Performance:** Optimized for performance, marimo executes only the necessary cells.
*   **Comprehensive Editor:** Features like Github Copilot, AI assistants, Ruff code formatting, and an interactive dataframe viewer.

### Explore the Possibilities

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Reactive Notebooks">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Interactive UI">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Data Visualization">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Layout Customization">
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
        <img src="https://marimo.io/shield.svg" alt="Playground Example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Playground Example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Playground Example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Playground Example"/>
      </a>
    </td>
  </tr>
</table>

##  Community and Resources

*   **Documentation:**  Dive deeper into the features and functionalities of marimo at [https://docs.marimo.io](https://docs.marimo.io).
*   **Examples:** Find practical examples to get you started at [https://docs.marimo.io/examples/](https://docs.marimo.io/examples/).
*   **Gallery:** Discover a variety of notebook applications at [https://marimo.io/gallery](https://marimo.io/gallery).
*   **Online Playground:** Try marimo directly in your browser: [https://marimo.app/l/c7h6pz](https://marimo.app/l/c7h6pz).
*   **YouTube Channel:** Learn more through tutorials and demos [https://www.youtube.com/@marimo-team](https://www.youtube.com/@marimo-team).

## Contributing

We welcome contributions!  Check out [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for guidelines.

## Join the Community

Connect with the marimo community for support, discussions, and updates.

*   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**marimo** is affiliated with the NumFOCUS community.
<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS Affiliated Project" />
```

Key improvements and SEO considerations:

*   **Strong Headline:**  Uses the keyword "Python Notebook" and emphasizes key benefits.
*   **Descriptive Subheadings:** Organizes content logically.
*   **Bulleted Feature List:** Highlights key functionalities for quick comprehension.
*   **Contextual Links:** Links to relevant documentation and external resources (with descriptive anchor text).
*   **Clear Call to Action:** Encourages users to explore the project, install, and contribute.
*   **Concise Language:** Avoids jargon and uses straightforward descriptions.
*   **Keywords in Text:** Strategically incorporates keywords like "Python notebook," "reactive," "data science," "interactive," and "git-friendly."
*   **Alt Text for Images:** Provides alt text for all images to improve accessibility and SEO.
*   **Community Section:**  Promotes user engagement.
*   **NumFOCUS Affiliation:** Reinforces credibility.