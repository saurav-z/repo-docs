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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"></a>
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads">
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" >
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

## **marimo: Revolutionizing Python Notebooks for Data Science**

marimo is a next-generation, reactive Python notebook that transforms how you work with data, offering reproducibility, Git-friendliness, and seamless deployment.  [Get started with marimo on GitHub](https://github.com/marimo-team/marimo)!

**Key Features:**

*   **🚀 Batteries-Included:** Replaces Jupyter, Streamlit, and more, simplifying your data science workflow.
*   **⚡️ Reactive Execution:** Automatically updates dependent cells when you modify code or interact with UI elements.
*   **🖐️ Interactive UI Elements:** Create and bind interactive sliders, tables, plots, and more directly to your Python code without callbacks.
*   **🐍 Git-Friendly:** Stores notebooks as `.py` files, making version control easy.
*   **🛢️ Data-Focused:** Work with dataframes, databases, warehouses, and lakehouses with SQL support.
*   **🤖 AI-Native:** Generate code and entire notebooks with AI, customized for data work.
*   **🔬 Reproducible & Deterministic:**  Eliminates hidden state and ensures consistent results.
*   **🏃 Executable Scripts:** Run your notebooks as standard Python scripts with CLI argument support.
*   **🛜 Shareable & Deployable:** Deploy notebooks as interactive web apps, slides, or run them in the browser using WASM.
*   **🧩 Reusable Code:** Import functions and classes between notebooks.
*   **🧪 Testable:** Integrate with `pytest` for robust testing.
*   **⌨️ Modern Editor:** Benefit from a modern code editor with features like GitHub Copilot, AI assistants, Vim keybindings, and a variable explorer.

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [quickstart](#quickstart) for a primer on our CLI._

## **Why marimo? A Reactive Programming Environment**

marimo offers a reactive programming environment that guarantees code, outputs, and program state consistency, solving many issues associated with traditional notebooks.

**Key Benefits:**

*   **Reactive Execution:**  Modify a cell and watch marimo automatically update dependent cells, ensuring your results are always up-to-date.  Delete a cell, and marimo scrubs its variables from program memory, eliminating hidden state.
    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Execution GIF" />

*   **Compatibility with Large Notebooks:** Configure marimo to mark cells as stale rather than automatically running them to prevent accidental execution of expensive computations.

*   **Synchronized UI Elements:** Easily integrate UI elements such as sliders, dropdowns, dataframe transformers, and chat interfaces. When you interact with these, the cells that use them are automatically re-run with their latest values.
    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI elements GIF" />

*   **Interactive DataFrames:**  Quickly page through, search, filter, and sort millions of rows directly within your notebooks.
    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Dataframes GIF" />

*   **AI-Powered Code Generation:**  Generate code with an AI assistant specialized for data work. Generate code for specific tasks, customize system prompts, and use local models.
    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI generation GIF" />

*   **SQL Queries:** Build SQL queries directly in your notebooks, dependent on Python variables, and execute them against various data sources.
    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL cell image" />

*   **Dynamic Markdown:** Create interactive and dynamic markdown content that depends on Python data.

*   **Built-in Package Management:** Seamlessly install and manage packages, including the ability to serialize and install dependencies in isolated environments.

*   **Deterministic Execution Order:** Your notebook's execution order is based on variable dependencies, giving you control over the flow of your code.

*   **Performant Runtime:** marimo optimizes execution by running only the necessary cells through static code analysis.

## **Quickstart: Get Started with marimo**

_The [marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq) on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an overview of many features._

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with additional dependencies that unlock SQL cells, AI completion, and more, run

```bash
pip install marimo[recommended]
```

**Key Commands:**

*   **Create/Edit Notebooks:**
    ```bash
    marimo edit
    ```

*   **Run as Web App:**
    ```bash
    marimo run your_notebook.py
    ```
    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Model comparison GIF" />

*   **Execute as Script:**
    ```bash
    python your_notebook.py
    ```

*   **Convert Jupyter Notebooks:**
    ```bash
    marimo convert your_notebook.ipynb > your_notebook.py
    ```
    or use our [web interface](https://marimo.io/convert).

*   **Tutorials:**
    ```bash
    marimo tutorial --help
    ```

## **Need Help?**

Check out the [FAQ](https://docs.marimo.io/faq.html) in our documentation for answers to common questions.

## **Learn More About marimo**

marimo offers a wealth of features for both beginners and advanced users.

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="marimo reactive execution">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="marimo UI elements">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="marimo intro">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="marimo layouts">
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
        <img src="https://marimo.io/shield.svg" alt="marimo playground example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="marimo playground example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="marimo playground example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="marimo playground example"/>
      </a>
    </td>
  </tr>
</table>

## **Contribute to marimo**

We welcome contributions from everyone!  See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details on how to get started.

> Questions? Reach out to us [on Discord](https://marimo.io/discord?ref=readme).

## **Join the marimo Community**

Connect with us and be part of the growing marimo community!

*   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**A NumFOCUS affiliated project.** marimo is a proud member of the NumFOCUS community, supporting projects such as NumPy, SciPy, and Matplotlib.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS Logo" />

## **Inspiration: The Future of Python Notebooks**

marimo is inspired by projects like Pluto.jl and ObservableHQ, aiming to reinvent the Python notebook as a reproducible, interactive, and shareable program.  We believe in empowering developers with the best tools, and we hope marimo transforms how you approach data science, research, and communication.