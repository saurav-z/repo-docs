<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="Marimo Logo">
</p>

<p align="center">
  <em>**Marimo: Build Interactive Python Notebooks That Are Reproducible, Shareable, and Scriptable**</em>
</p>

<p align="center">
  <a href="https://docs.marimo.io" target="_blank"><strong>Docs</strong></a> ·
  <a href="https://marimo.io/discord?ref=readme" target="_blank"><strong>Discord</strong></a> ·
  <a href="https://docs.marimo.io/examples/" target="_blank"><strong>Examples</strong></a> ·
  <a href="https://marimo.io/gallery/" target="_blank"><strong>Gallery</strong></a> ·
  <a href="https://www.youtube.com/@marimo-team/" target="_blank"><strong>YouTube</strong></a>
  <br>
  <a href="https://github.com/marimo-team/marimo"><strong>GitHub Repo</strong></a>
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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI Version"></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda Version"></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"/></a>
</p>

## What is Marimo?

Marimo is a modern, reactive Python notebook designed for data scientists, researchers, and developers. It transforms your Python code into interactive, reproducible, and shareable experiences, offering a superior alternative to traditional notebooks.

## Key Features

*   🚀 **Batteries-Included:** Replaces multiple tools like Jupyter, Streamlit, and more, streamlining your workflow.
*   ⚡️ **Reactive Programming:** Automatically updates dependent cells when a cell's code or a UI element changes, ensuring consistency ([Learn more](https://docs.marimo.io/guides/reactivity.html)).
*   🖐️ **Interactive UI Elements:** Integrate sliders, tables, plots, and other interactive elements directly into your notebooks without callbacks ([Learn more](https://docs.marimo.io/guides/interactivity.html)).
*   🐍 **Git-Friendly:** Saves notebooks as `.py` files for easy version control and collaboration.
*   🛢️ **Data-Focused:** Seamlessly works with dataframes, databases, SQL, and other data sources ([Learn more](https://docs.marimo.io/guides/working_with_data/sql.html)).
*   🤖 **AI-Enhanced:** Generate and refine code with AI assistants tailored for data science ([Learn more](https://docs.marimo.io/guides/generate_with_ai/)).
*   🔬 **Reproducible:** Ensures consistent results with no hidden state and built-in package management ([Learn more](https://docs.marimo.io/guides/reactivity.html#no-hidden-state)).
*   🏃 **Executable as Scripts:** Run your notebooks as standard Python scripts, including with CLI arguments.
*   🛜 **Shareable & Deployable:** Easily deploy notebooks as interactive web apps or create presentations (slides) ([Learn more](https://docs.marimo.io/guides/apps.html)).
*   🧩 **Reusable Code:** Import and reuse functions and classes between notebooks.
*   🧪 **Testable:** Integrate with pytest for robust testing.
*   ⌨️ **Modern Editor:** Includes GitHub Copilot, AI assistance, Vim keybindings, a variable explorer, and many other features ([Learn more](https://docs.marimo.io/guides/editor_features/index.html)).

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [quickstart](#quickstart) for a primer on our CLI._

## How Marimo Works: A Reactive Programming Environment

Marimo guarantees consistency between your code, outputs, and program state. This contrasts with traditional notebooks like Jupyter, where the relationship between code and outputs can be fragile.

**Key Benefits:**

*   **Reactive Updates:** When you modify a cell, marimo automatically re-runs any cells that depend on its output, keeping everything in sync. Deleting a cell removes its variables from program memory, eliminating hidden state.
    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Example">

*   **Optimized for Large Notebooks:** Configure marimo's runtime to "lazily" update cells, marking them as stale instead of immediately running them to prevent accidental execution of expensive computations ([Learn more](https://docs.marimo.io/guides/configuration/runtime_configuration.html)).

*   **UI Element Synchronization:** Interactive elements like sliders, dropdowns, and dataframes automatically update the cells that use their values.
    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI Element Example">

*   **Fast Dataframe Interaction:** Effortlessly page, search, filter, and sort through millions of rows of data directly within your notebook.
    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-df.gif" width="700px" alt="Dataframe Example">

*   **AI-Powered Code Generation:** Utilize an AI assistant to generate data-aware code tailored to your variables in memory, or create entire notebooks from text descriptions ([Learn more](https://docs.marimo.io/guides/generate_with_ai/)).
    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-generate-with-ai.gif" width="700px" alt="AI-Powered Example">

*   **SQL Integration:** Build SQL queries that depend on Python values and execute them against a variety of data sources (dataframes, databases, etc.) with our built-in SQL engine.
    <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-sql-cell.png" width="700px" alt="SQL Example">

## Quickstart

_The [marimo concepts playlist](https://www.youtube.com/watch?v=3N6lInzq5MI&list=PLNJXGo8e1XT9jP7gPbRdm1XwloZVFvLEq) on our [YouTube channel](https://www.youtube.com/@marimo-team) gives an overview of many features._

**Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with all dependencies (SQL support, AI, etc.), run:

```bash
pip install marimo[recommended]
```

**Create/Edit Notebooks:**

```bash
marimo edit
```

**Run as Web App:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Web App Example">

**Execute as Script:**

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

**Cloud-Based Notebooks:** Use [molab](https://molab.marimo.io/notebooks) for cloud-based collaboration.

## Support and Resources

*   **FAQ:** [https://docs.marimo.io/faq.html](https://docs.marimo.io/faq.html)

## Learn More & Explore

Marimo offers a wide range of capabilities for both beginners and advanced users.

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

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for details.

> Questions?  Join us on [Discord](https://marimo.io/discord?ref=readme).

## Community

Join the Marimo community!

*   ⭐ [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**NumFOCUS Affiliated Project:**  Marimo is part of the NumFOCUS community, which includes projects such as NumPy, SciPy, and Matplotlib.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS">

## Inspiration

Marimo aims to reinvent Python notebooks as shareable and reproducible programs.  We draw inspiration from projects like Pluto.jl and ObservableHQ, and the ideas of reactive dataflow programming are transforming the tools we use.

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="Marimo Logo">
</p>
```

Key improvements and SEO considerations:

*   **Clear Title & Hook:**  The initial sentence clearly states what Marimo is.  The use of keywords "Python notebook," "reactive," "reproducible," and "shareable" is strategic.
*   **Target Keywords:**  The text is filled with relevant keywords: "Python notebook," "reactive," "interactive," "data science," "reproducible," "shareable," "scriptable," "dataframes," "SQL," "AI," "web app," "Jupyter," "Streamlit," and more.
*   **Organized Structure:** The use of headings and subheadings makes the README easy to read and scan.
*   **Bulleted Key Features:**  This is a best practice for highlighting features.
*   **Visuals:**  The use of images (GIFs and screenshots) enhances the README and makes it more engaging.  `alt` tags are included for accessibility.
*   **Call to Action:** The `pip install` command and the link to the online playground give readers a clear next step.
*   **Internal Linking:** Linking to different sections of the README and the documentation increases the user's time on the page.
*   **SEO-Friendly Language:**  The text uses clear, concise language optimized for search engines.
*   **Community and Social Links:**  Promotes engagement and provides links to social media platforms.
*   **NumFOCUS Affiliation:** Adds credibility and authority.
*   **Concise and Focused:**  The text avoids unnecessary jargon and keeps the focus on the core benefits of Marimo.
*   **GitHub Repo Link:** Ensures users can easily find the project's repository.