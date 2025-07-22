<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="marimo logo">
</p>

<h1 align="center">marimo: The Reactive Python Notebook</h1>

<p align="center">
  <em>Create reproducible, interactive, and shareable Python programs with marimo.</em>
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
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads" alt="PyPI Downloads">
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" alt="Conda Downloads">
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"></a>
</p>

**marimo** is a next-generation Python notebook that empowers data scientists and developers to create, share, and deploy interactive and reproducible Python applications. Visit the [GitHub repository](https://github.com/marimo-team/marimo) to learn more.

**Key Features:**

*   🚀 **Batteries-included:** Combines features of `jupyter`, `streamlit`, `jupytext`, `ipywidgets`, and more.
*   ⚡️ **Reactive Programming:** Automatically runs dependent cells when a cell is modified, ensuring consistency.
*   🖐️ **Interactive UIs:** Easily create interactive elements like sliders, tables, and plots that directly connect to your Python code.
*   🐍 **Git-Friendly:** Notebooks are stored as `.py` files for easy version control.
*   🛢️ **Data-Focused:** Seamlessly work with data using SQL queries, dataframe manipulation, and more.
*   🤖 **AI-Enhanced:** Generate code with AI assistants specifically designed for data work.
*   🔬 **Reproducible:** Guarantees deterministic execution and no hidden state.
*   🏃 **Executable as Scripts:** Run your notebooks as standard Python scripts.
*   🛜 **Shareable & Deployable:** Deploy interactive web apps, slides, and more.
*   🧩 **Reusable Components:** Import functions and classes between notebooks.
*   🧪 **Testable:** Integrate with `pytest` for robust testing.
*   ⌨️ **Modern Editor:** Includes features like GitHub Copilot, AI assistants, Vim keybindings, and a variable explorer.

```bash
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz) for an in-browser experience!_

## Why Use marimo?

marimo addresses the limitations of traditional notebooks by providing a reactive programming environment that ensures code, outputs, and program state are always consistent. This leads to more reliable and maintainable data science workflows.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive demo">

**Key Benefits:**

*   **Reactive Execution:** Automatically updates dependent cells, eliminating manual re-runs.
*   **Flexible Runtime:** Configure lazy execution for expensive notebooks.
*   **Synchronized UI Elements:** Interactive UI elements update cells automatically.
*   **Interactive Dataframes:** Quickly page, search, filter, and sort large datasets.
*   **AI-Powered Code Generation:** Utilize AI assistants tailored for data work.
*   **SQL Integration:** Query data directly from various sources using SQL.
*   **Dynamic Markdown:** Create dynamic narratives using Python variables in markdown.
*   **Built-in Package Management:** Easily manage dependencies.
*   **Deterministic Execution:** Notebooks execute in a predictable order.
*   **Performance:** Efficiently runs only the necessary cells.
*   **All-in-One:** Offers an extensive set of features.

## Quickstart

_See the [marimo concepts
playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq)
on our [YouTube channel](https://www.youtube.com/@marimo-team) for an overview of its key features._

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

For extended features:

```bash
pip install marimo[recommended]
```

**Workflow:**

1.  **Create or Edit:**
    ```bash
    marimo edit
    ```
2.  **Run as Web App:**
    ```bash
    marimo run your_notebook.py
    ```
    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Marimo demo">
3.  **Execute as Script:**
    ```bash
    python your_notebook.py
    ```
4.  **Convert Jupyter Notebooks:**
    ```bash
    marimo convert your_notebook.ipynb > your_notebook.py
    ```
    or use our [web interface](https://marimo.io/convert).
5.  **Explore Tutorials:**
    ```bash
    marimo tutorial --help
    ```

## Resources

*   [Documentation](https://docs.marimo.io)
*   [Examples](https://docs.marimo.io/examples/)
*   [Gallery](https://marimo.io/gallery)

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Tutorial">
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
        <img src="https://marimo.io/shield.svg" alt="Playground">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/0ue871">
        <img src="https://marimo.io/shield.svg" alt="Example">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/lxp1jk">
        <img src="https://marimo.io/shield.svg" alt="Example">
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.app/l/14ovyr">
        <img src="https://marimo.io/shield.svg" alt="Example">
      </a>
    </td>
  </tr>
</table>

## Contributing

Contribute to marimo's development by following the guidelines in [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md).

> Need help?  Join us [on Discord](https://marimo.io/discord?ref=readme).

## Community

Join the marimo community and stay updated:

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

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS logo" />

## Inspiration

marimo is inspired by projects like Pluto.jl, ObservableHQ, and Bret Victor's essays. It's part of a broader movement towards reactive dataflow programming.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo logo horizontal">
</p>
```
Key improvements and explanations:

*   **SEO-Optimized Title:**  Includes "Python notebook" and relevant keywords in the title and headings.
*   **Concise Hook:** Starts with a one-sentence description that summarizes the core value proposition.
*   **Clear Headings:** Uses headings (H1, H2) to structure the content logically, making it easy to scan.
*   **Bulleted Key Features:** Uses bullet points to highlight the core functionality.  This improves readability and helps users quickly grasp the key benefits.  The features are more concisely described.
*   **Focus on User Benefits:** The description focuses on *what the user gets* (e.g., "easily create interactive elements") rather than just the technical features.
*   **Stronger Call to Action:**  Includes clear calls to action (e.g., "Join the marimo community and stay updated").
*   **Clearer Quickstart:** Streamlined the quickstart instructions.
*   **Alt Text for Images:** Added `alt` text to all images for accessibility and SEO.
*   **Concise Language:** Improved the overall readability by shortening sentences and using more direct language.
*   **Removes Redundancy:**  Removed redundant information and phrases.
*   **Improved Formatting:** Consistent formatting for links and headings.
*   **Inclusion of Keywords:**  Incorporates relevant keywords throughout the text naturally.
*   **Community Section:**  Highlights how to connect with the community.
*   **NumFOCUS Affiliation:** Clearly indicates NumFOCUS affiliation.
*   **Inspiration Section:**  Highlights the inspiration behind marimo.

This improved README is more likely to attract users, be indexed by search engines, and effectively communicate the value of marimo.