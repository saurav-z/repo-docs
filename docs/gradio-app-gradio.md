<div align="center">
<a href="https://gradio.app">
<img src="readme_files/gradio.svg" alt="gradio" width=350>
</a>
</div>

<div align="center">
<span>
<a href="https://www.producthunt.com/posts/gradio-5-0?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-gradio&#0045;5&#0045;0" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=501906&theme=light" alt="Gradio&#0032;5&#0046;0 - the&#0032;easiest&#0032;way&#0032;to&#0032;build&#0032;AI&#0032;web&#0032;apps | Product Hunt" style="width: 150px; height: 54px;" width="150" height="54" /></a>
<a href="https://trendshift.io/repositories/2145" target="_blank"><img src="https://trendshift.io/api/badge/repositories/2145" alt="gradio-app%2Fgradio | Trendshift" style="width: 150px; height: 55px;" width="150" height="55"/></a>
</span>

[![gradio-backend](https://github.com/gradio-app/gradio/actions/workflows/test-python.yml/badge.svg)](https://github.com/gradio-app/gradio/actions/workflows/test-python.yml)
[![gradio-ui](https://github.com/gradio-app/gradio/actions/workflows/tests-js.yml/badge.svg)](https://github.com/gradio-app/gradio/actions/workflows/tests-js.yml) 
[![PyPI](https://img.shields.io/pypi/v/gradio)](https://pypi.org/project/gradio/)
[![PyPI downloads](https://img.shields.io/pypi/dm/gradio)](https://pypi.org/project/gradio/)
![Python version](https://img.shields.io/badge/python-3.10+-important)
[![Twitter follow](https://img.shields.io/twitter/follow/gradio?style=social&label=follow)](https://twitter.com/gradio)

[Website](https://gradio.app)
| [Documentation](https://gradio.app/docs/)
| [Guides](https://gradio.app/guides/)
| [Getting Started](https://gradio.app/getting_started/)
| [Examples](demo/)

</div>

<div align="center">

English | [中文](readme_files/zh-cn#readme)

</div>

# Gradio: Build and Share Machine Learning Demos with Ease 🚀

**Gradio empowers you to create interactive web applications for your machine learning models and APIs in minutes, with no front-end experience required.**

## Key Features

*   **Rapid Prototyping:** Quickly build demos for your models and APIs using Python.
*   **Easy Sharing:** Share your demos with a public URL with the `share=True` parameter in `launch()`.
*   **Interactive Components:** Utilize a wide range of pre-built components for inputs and outputs (textboxes, images, audio, and more).
*   **Customization:** Leverage `gr.Blocks` for greater layout control and data flow design.
*   **Chatbot Creation:** Create chatbots easily with `gr.ChatInterface`.
*   **Extensive Ecosystem:** Access Python and JavaScript clients for programmatic interaction, Gradio-Lite for in-browser apps, and seamless integration with Hugging Face Spaces.
*   **Hot Reload Mode:** Run your Gradio app in hot reload mode.

## Getting Started

### Installation

**Prerequisite**: Gradio requires [Python 3.10 or higher](https://www.python.org/downloads/).

Install Gradio using `pip`:

```bash
pip install --upgrade gradio
```

> [!TIP]
 > It is best to install Gradio in a virtual environment. Detailed installation instructions for all common operating systems <a href="https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment">are provided here</a>.

### Build Your First Demo

Create a basic "Hello World" demo in Python:

```python
import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()
```

Run your Python file (e.g., `python app.py`).  Your demo will launch in your browser.

> [!TIP]
 > When developing locally, you can run your Gradio app in <strong>hot reload mode</strong>, which automatically reloads the Gradio app whenever you make changes to the file. To do this, simply type in <code>gradio</code> before the name of the file instead of <code>python</code>. In the example above, you would type: `gradio app.py` in your terminal. You can also enable <strong>vibe mode</strong> by using the <code>--vibe</code> flag, e.g. <code>gradio --vibe app.py</code>, which provides an in-browser chat that can be used to write or edit your Gradio app using natural language. Learn more in the <a href="https://www.gradio.app/guides/developing-faster-with-reload-mode">Hot Reloading Guide</a>.

## Core Concepts

*   **`gr.Interface`:**  The primary class for creating simple demos; wraps a Python function with a UI.  Takes `fn` (the function), `inputs` (input components), and `outputs` (output components) as arguments.
*   **Components:** Gradio offers a variety of built-in components (e.g., `gr.Textbox`, `gr.Image`) for different input and output types.
*   **Sharing:**  Use `demo.launch(share=True)` to generate a public URL for your demo.
*   **`gr.Blocks`:** For more advanced layouts and control over data flow.
*   **`gr.ChatInterface`:** Create chatbot UIs quickly.

## Diving Deeper

*   **Gradio Guides:**  Learn sequentially through the [Gradio Guides](https://www.gradio.app/guides/).
*   **API Documentation:**  Explore the [technical API documentation](https://www.gradio.app/docs/).

## Gradio Sketch

You can also build Gradio applications without writing any code. Simply type `gradio sketch` into your terminal to open up an editor that lets you define and modify Gradio components, adjust their layouts, add events, all through a web editor. Or [use this hosted version of Gradio Sketch, running on Hugging Face Spaces](https://huggingface.co/spaces/aliabid94/Sketch).

## Questions?

Report bugs/feature requests on [GitHub](https://github.com/gradio-app/gradio/issues/new/choose) or ask questions on [Discord](https://discord.com/invite/feTf9x3ZSB).

If you like Gradio, please leave us a ⭐ on GitHub!  [Back to Top](#gradio-build-and-share-machine-learning-demos-with-ease-)

## Open Source Stack

Gradio is built on top of many wonderful open-source libraries!

[<img src="readme_files/huggingface_mini.svg" alt="huggingface" height=40>](https://huggingface.co)
[<img src="readme_files/python.svg" alt="python" height=40>](https://www.python.org)
[<img src="readme_files/fastapi.svg" alt="fastapi" height=40>](https://fastapi.tiangolo.com)
[<img src="readme_files/encode.svg" alt="encode" height=40>](https://www.encode.io)
[<img src="readme_files/svelte.svg" alt="svelte" height=40>](https://svelte.dev)
[<img src="readme_files/vite.svg" alt="vite" height=40>](https://vitejs.dev)
[<img src="readme_files/pnpm.svg" alt="pnpm" height=40>](https://pnpm.io)
[<img src="readme_files/tailwind.svg" alt="tailwind" height=40>](https://tailwindcss.com)
[<img src="readme_files/storybook.svg" alt="storybook" height=40>](https://storybook.js.org/)
[<img src="readme_files/chromatic.svg" alt="chromatic" height=40>](https://www.chromatic.com/)

## License

Gradio is licensed under the Apache License 2.0 found in the [LICENSE](LICENSE) file in the root directory of this repository.

## Citation

Also check out the paper _[Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild](https://arxiv.org/abs/1906.02569), ICML HILL 2019_, and please cite it if you use Gradio in your work.

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}
```

[Back to Top](#gradio-build-and-share-machine-learning-demos-with-ease-)
```

Key changes and improvements:

*   **SEO Optimization:**  Added keywords like "machine learning," "web apps," "demos," "Python," and "interactive."
*   **Concise Hook:**  The one-sentence hook is at the top to grab attention.
*   **Clear Headings:**  Improved the structure with clear headings and subheadings.
*   **Bulleted Key Features:**  Highlights the benefits concisely.
*   **Action-Oriented Language:**  Uses verbs like "build," "share," "create," and "explore" to encourage engagement.
*   **Simplified Formatting:** Improved readability with bold text for emphasis.
*   **Call to Action:**  Encourages users to get started and explore further.
*   **Added [Back to Top](#gradio-build-and-share-machine-learning-demos-with-ease-) link** to the end of sections.
*   **Cleaned up formatting** to improve readability.
*   **Removed unnecessary repetition.**