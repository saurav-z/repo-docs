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

# Gradio: Build Machine Learning Web Apps with Ease

**Gradio empowers you to rapidly create user-friendly web interfaces for your machine learning models and Python functions, all with minimal code and no frontend experience required!**

[Visit the original repo](https://github.com/gradio-app/gradio)

## Key Features

*   **Simple and Fast:** Build demos and web apps for your ML models in minutes with just a few lines of Python.
*   **No Frontend Required:**  No need to learn HTML, CSS, or JavaScript. Gradio handles the UI for you.
*   **Shareable Demos:** Easily share your apps with a public URL with the `share=True` parameter.
*   **Rich Component Library:**  Utilize a wide range of built-in UI components for various input and output types (text, images, audio, etc.).
*   **Customizable Layouts:**  Leverage `gr.Blocks` for advanced layouts and data flow control to create complex applications.
*   **Chatbot Support:**  Quickly build chatbots with `gr.ChatInterface`.
*   **Ecosystem of Tools:** Integrate with Python and JavaScript clients, and explore browser-based Gradio-Lite and hosted solutions like Hugging Face Spaces.

## Installation

**Prerequisites:** Python 3.10 or higher.

Install Gradio using pip:

```bash
pip install --upgrade gradio
```
> [!TIP]
 > It is best to install Gradio in a virtual environment. Detailed installation instructions for all common operating systems <a href="https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment">are provided here</a>. 

## Getting Started

Here's a basic example to get you started:

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

Run this code, and Gradio will generate a web interface for your Python function. You can then interact with it in your browser.

> [!TIP]
 > When developing locally, you can run your Gradio app in <strong>hot reload mode</strong>, which automatically reloads the Gradio app whenever you make changes to the file. To do this, simply type in <code>gradio</code> before the name of the file instead of <code>python</code>. In the example above, you would type: `gradio app.py` in your terminal. You can also enable <strong>vibe mode</strong> by using the <code>--vibe</code> flag, e.g. <code>gradio --vibe app.py</code>, which provides an in-browser chat that can be used to write or edit your Gradio app using natural language. Learn more in the <a href="https://www.gradio.app/guides/developing-faster-with-reload-mode">Hot Reloading Guide</a>.

### Understanding the `Interface` Class

The `gr.Interface` class is your primary tool for creating demos. It takes three core arguments:

*   `fn`: Your Python function.
*   `inputs`:  The input Gradio components (e.g., text, image).
*   `outputs`: The output Gradio components.

## Sharing Your Demo

Easily share your demo with the world by setting `share=True` in the `launch()` method:

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
```

## Gradio: Under the Hood

Beyond `gr.Interface`, Gradio offers:

*   **`gr.Blocks`:** For building complex, customizable apps with control over layout and data flow.
*   **`gr.ChatInterface`:** For quick and easy chatbot creation.
*   **Python and JavaScript Ecosystem:** Gradio client libraries for interacting with apps programmatically.

## Resources

*   [Gradio Website](https://gradio.app)
*   [Documentation](https://gradio.app/docs/)
*   [Guides](https://gradio.app/guides/)
*   [Examples](demo/)
*   [Gradio Sketch](https://huggingface.co/spaces/aliabid94/Sketch)

## Questions & Support

*   [Report Issues on GitHub](https://github.com/gradio-app/gradio/issues/new/choose)
*   [Join our Discord Server](https://discord.com/invite/feTf9x3ZSB)

If you find Gradio useful, please consider leaving a ⭐ on GitHub!

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

```
@article{abid2019gradio,
  title = {Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild},
  author = {Abid, Abubakar and Abdalla, Ali and Abid, Ali and Khan, Dawood and Alfozan, Abdulrahman and Zou, James},
  journal = {arXiv preprint arXiv:1906.02569},
  year = {2019},
}