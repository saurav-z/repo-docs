<p align="center">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-thick.svg" alt="marimo logo">
</p>

<h1 align="center">marimo: Reactive Python Notebooks for Data Science</h1>

<p align="center">
  <em>Transform your data science workflow with marimo, a powerful Python notebook that's reproducible, git-friendly, and deployable as scripts or apps.</em>
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
<a href="https://pypi.org/project/marimo/"><img src="https://img.shields.io/pypi/v/marimo?color=%2334D058&label=pypi" alt="PyPI version"/></a>
<a href="https://anaconda.org/conda-forge/marimo"><img src="https://img.shields.io/conda/vn/conda-forge/marimo.svg" alt="Conda version"/></a>
<a href="https://marimo.io/discord?ref=readme"><img src="https://shields.io/discord/1059888774789730424" alt="Discord"/></a>
<img alt="PyPI Downloads" src="https://img.shields.io/pepy/dt/marimo?label=pypi%20%7C%20downloads"/>
<img alt="Conda Downloads" src="https://img.shields.io/conda/d/conda-forge/marimo" />
<a href="https://github.com/marimo-team/marimo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/marimo" alt="License"/></a>
</p>

## Key Features of marimo

marimo is a modern Python notebook designed for data scientists and engineers, offering a superior development experience. 

*   🚀 **All-in-One Solution:** Replaces Jupyter, Streamlit, and more with a single, streamlined tool.
*   ⚡️ **Reactive Programming:** Automatically updates dependent cells when a cell's input changes, ensuring code and outputs stay consistent. [Learn More](https://docs.marimo.io/guides/reactivity.html)
*   🖐️ **Interactive UI Elements:** Easily bind sliders, tables, plots, and more to your Python code without callbacks. [Explore Interactivity](https://docs.marimo.io/guides/interactivity.html)
*   🐍 **Git-Friendly:** Store notebooks as pure `.py` files for seamless version control and collaboration.
*   🛢️ **Data-Focused:** Work with data efficiently using SQL queries, dataframe manipulation, and more.  [SQL Support](https://docs.marimo.io/guides/working_with_data/sql.html)
*   🤖 **AI-Enhanced:** Generate cells with AI, tailored for data work. [AI Integration](https://docs.marimo.io/guides/generate_with_ai/)
*   🔬 **Reproducible Results:** Deterministic execution, no hidden state, and built-in package management ensure reproducible results.
*   🏃 **Executable as Scripts:** Execute your notebooks as standard Python scripts with CLI arguments.
*   🛜 **Shareable & Deployable:** Deploy notebooks as interactive web apps, slides, or run them in the browser.
*   🧩 **Reusable Code:** Import functions and classes from one notebook to another for modularity.
*   🧪 **Testable Code:**  Integrate with pytest for robust testing of your notebooks.
*   ⌨️ **Modern Editor:** Includes GitHub Copilot, AI assistants, Vim keybindings, a variable explorer, and more to enhance your workflow. [Editor Features](https://docs.marimo.io/guides/editor_features/index.html)

```python
pip install marimo && marimo tutorial intro
```

_Try marimo at [our online playground](https://marimo.app/l/c7h6pz), which runs entirely in the browser!_

_Jump to the [Quickstart](#quickstart) for a primer on our CLI._

## What is marimo?

marimo is a revolutionary reactive Python notebook that allows you to build reproducible, shareable, and interactive data science projects; it provides a robust and intuitive platform for data exploration, analysis, and presentation. Unlike traditional notebooks, marimo ensures that your code, outputs, and program state are always consistent, reducing errors and streamlining your workflow. Check out the [original repo](https://github.com/marimo-team/marimo) for more information.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/reactive.gif" width="700px" alt="Reactive Example GIF" />

### Key Benefits

*   **Eliminate Hidden State:**  marimo's reactive nature ensures that changes in one cell automatically update dependent cells, minimizing errors.
*   **UI Element Synchronization:**  Easily integrate UI elements like sliders, dropdowns, and interactive dataframes.
*   **Interactive Dataframes:**  Page through, search, filter, and sort millions of rows with ease.
*   **AI-Powered Code Generation:** Leverage AI assistants to generate code tailored for data analysis tasks.
*   **Built-in SQL Engine:** Seamlessly query data from various sources using SQL within your notebooks.
*   **Dynamic Markdown:** Create dynamic and engaging narratives with markdown that responds to Python data.
*   **Package Management:** Easily manage dependencies with built-in package management.
*   **Deterministic Execution Order:** Organize your notebooks logically, independent of cell order.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" width="700px" alt="UI Elements Example GIF" />

## Quickstart

Ready to get started?  Here's how to begin:

**1. Installation:**

```bash
pip install marimo  # or conda install -c conda-forge marimo
marimo tutorial intro
```

To install with extra dependencies for extended features (SQL, AI, etc.):

```bash
pip install marimo[recommended]
```

**2. Create/Edit Notebooks:**

```bash
marimo edit
```

**3. Run as a Web App:**

```bash
marimo run your_notebook.py
```

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-model-comparison.gif" style="border-radius: 8px" width="450px" alt="Web App Example" />

**4. Execute as a Script:**

```bash
python your_notebook.py
```

**5. Convert Jupyter Notebooks:**

```bash
marimo convert your_notebook.ipynb > your_notebook.py
```

Or use the [web interface](https://marimo.io/convert).

**6. Tutorials:**

```bash
marimo tutorial --help
```

## Learn More

marimo offers a wealth of features for both beginners and advanced users.

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="https://docs.marimo.io/getting_started/key_concepts.html">
        <img src="https://docs.marimo.io/_static/reactive.gif" style="max-height: 150px; width: auto; display: block" alt="Tutorial Example"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/inputs/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/readme-ui.gif" style="max-height: 150px; width: auto; display: block" alt="Inputs Example" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/guides/working_with_data/plotting.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/docs-intro.gif" style="max-height: 150px; width: auto; display: block" alt="Plots Example" />
      </a>
    </td>
    <td>
      <a target="_blank" href="https://docs.marimo.io/api/layouts/index.html">
        <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/outputs.gif" style="max-height: 150px; width: auto; display: block" alt="Layout Example" />
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

Explore more in our [docs](https://docs.marimo.io), [usage examples](https://docs.marimo.io/examples/), and [gallery](https://marimo.io/gallery).

## Contributing

We welcome contributions from the community! Please see [CONTRIBUTING.md](https://github.com/marimo-team/marimo/blob/main/CONTRIBUTING.md) for guidance.

> Need help?  Join us [on Discord](https://marimo.io/discord?ref=readme).

## Community & Resources

Stay connected and join our growing community:

*   🌟 [Star us on GitHub](https://github.com/marimo-team/marimo)
*   💬 [Chat with us on Discord](https://marimo.io/discord?ref=readme)
*   📧 [Subscribe to our Newsletter](https://marimo.io/newsletter)
*   ☁️ [Join our Cloud Waitlist](https://marimo.io/cloud)
*   ✏️ [Start a GitHub Discussion](https://github.com/marimo-team/marimo/discussions)
*   🦋 [Follow us on Bluesky](https://bsky.app/profile/marimo.io)
*   🐦 [Follow us on Twitter](https://twitter.com/marimo_io)
*   🎥 [Subscribe on YouTube](https://www.youtube.com/@marimo-team)
*   🕴️ [Follow us on LinkedIn](https://www.linkedin.com/company/marimo-io)

**NumFOCUS Affiliated Project:**  marimo is proud to be a part of the NumFOCUS community, supporting open-source scientific computing.

<img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/numfocus_affiliated_project.png" height="40px" alt="NumFOCUS logo" />

## Inspiration

marimo reimagines the Python notebook, focusing on reproducibility, interactivity, and shareability, creating a superior environment for research and communication.

We are inspired by projects like [Pluto.jl](https://github.com/fonsp/Pluto.jl), [ObservableHQ](https://observablehq.com/tutorials), and the work of [Bret Victor](http://worrydream.com/). marimo contributes to the broader movement toward reactive dataflow programming, influencing and drawing inspiration from projects like [IPyflow](https://github.com/ipyflow/ipyflow), [streamlit](https://github.com/streamlit/streamlit), [TensorFlow](https://github.com/tensorflow/tensorflow), [PyTorch](https://github.com/pytorch/pytorch/tree/main), [JAX](https://github.com/google/jax), and [React](https://github.com/facebook/react).

<p align="right">
  <img src="https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-logotype-horizontal.png" height="200px" alt="marimo horizontal logo">
</p>
```
Key improvements and optimization notes:

*   **SEO Keywords:** Added relevant keywords like "Python notebook," "reactive programming," "data science," "interactive," "reproducible," "Git-friendly," and "deployable."
*   **Clear Headings:** Organized the content with clear, descriptive headings (e.g., "Key Features," "What is marimo?").
*   **Concise Language:**  Streamlined the descriptions for better readability.
*   **Bulleted Lists:**  Used bullet points to highlight key features and benefits, making them easy to scan.
*   **Strong Introduction:**  Crafted a compelling one-sentence introduction to capture attention.
*   **Call to Actions:** Added clear calls to action (e.g., "Learn More," "Quickstart").
*   **Alt Text for Images:** Added `alt` text to all images for accessibility and SEO.
*   **Internal Links:**  Integrated internal links to different sections of the README.
*   **Removed Redundancy**: Eliminated duplicated information and consolidated similar content.
*   **Focus on Benefits:** Emphasized the benefits of using marimo rather than just listing features.
*   **Conciseness**:  Kept the content focused on the essential information to avoid overwhelming the reader.