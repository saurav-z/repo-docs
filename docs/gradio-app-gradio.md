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

# Gradio: Build AI Web Apps and ML Demos in Minutes

Gradio is the open-source Python library that empowers you to rapidly create user-friendly web interfaces and shareable demos for your machine learning models, APIs, and Python functions, with *no* frontend experience required.  [Visit the Gradio Repository on GitHub](https://github.com/gradio-app/gradio).

## Key Features

*   **Rapid Prototyping:** Quickly turn your Python functions and models into interactive demos with minimal code.
*   **Shareable Demos:** Easily share your demos with a shareable link or embed them on websites, eliminating the need for web hosting.
*   **Interactive Components:** Utilize a wide range of pre-built UI components like text boxes, sliders, image viewers, and more, designed for machine learning applications.
*   **Customization:** Leverage `gr.Blocks` to create custom layouts and data flows for complex interactions.
*   **Chatbot Creation:**  Build chatbot interfaces easily with the `gr.ChatInterface` class.
*   **Ecosystem Support:** Integrates seamlessly with the Hugging Face ecosystem and offers Python and JavaScript clients for programmatic access.
*   **In-Browser Development:** Write and edit Gradio apps directly in your browser with Gradio-Lite.
*   **No-Code Solution**: Build Gradio apps without writing any code with Gradio Sketch.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-guides/gif-version.gif" style="padding-bottom: 10px">

## Installation

**Prerequisite**: Gradio requires [Python 3.10 or higher](https://www.python.org/downloads/).

Install Gradio using `pip`:

```bash
pip install --upgrade gradio
```

> [!TIP]
> It is best to install Gradio in a virtual environment. Detailed installation instructions for all common operating systems <a href="https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment">are provided here</a>.

## Getting Started

Here's how to build your first Gradio app:

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

Run this code (e.g., `python app.py`) and your demo will be accessible in your browser at `http://localhost:7860`.

> [!TIP]
> When developing locally, you can run your Gradio app in <strong>hot reload mode</strong>, which automatically reloads the Gradio app whenever you make changes to the file. To do this, simply type in <code>gradio</code> before the name of the file instead of <code>python</code>. In the example above, you would type: `gradio app.py` in your terminal. You can also enable <strong>vibe mode</strong> by using the <code>--vibe</code> flag, e.g. <code>gradio --vibe app.py</code>, which provides an in-browser chat that can be used to write or edit your Gradio app using natural language. Learn more in the <a href="https://www.gradio.app/guides/developing-faster-with-reload-mode">Hot Reloading Guide</a>.

## Core Classes

### `gr.Interface`
This class is the foundation for building simple demos. It takes your function, input components, and output components to create a UI.

### `gr.Blocks`

For more advanced layouts and custom interactions, use `gr.Blocks`. This class gives you greater control over UI design and data flow.

### `gr.ChatInterface`
Create chatbot interfaces quickly using this dedicated class.

## Sharing Your Demo

Share your demo with the world by adding `share=True` to the `launch()` method:

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
```

This generates a public URL (e.g., `https://a23dsf231adb.gradio.live`) for your demo.

## The Gradio Ecosystem

Gradio includes a variety of tools to suit your needs:

*   **Gradio Python Client** (`gradio_client`): Programmatically query Gradio apps in Python.
*   **Gradio JavaScript Client** (`@gradio/client`): Programmatically query Gradio apps in JavaScript.
*   **Gradio-Lite** (`@gradio/lite`): Write Gradio apps that run entirely in the browser (no server required!).
*   **Hugging Face Spaces**: Host your Gradio applications for free.

## Next Steps

*   Explore the [Gradio Guides](https://www.gradio.app/guides/) for tutorials and examples.
*   Consult the [API Documentation](https://www.gradio.app/docs/) for detailed information.

## Gradio Sketch
You can also build Gradio applications without writing any code. Simply type `gradio sketch` into your terminal to open up an editor that lets you define and modify Gradio components, adjust their layouts, add events, all through a web editor. Or [use this hosted version of Gradio Sketch, running on Hugging Face Spaces](https://huggingface.co/spaces/aliabid94/Sketch).

## Get in Touch

*   Report bugs and request features on [GitHub](https://github.com/gradio-app/gradio/issues/new/choose).
*   Join the conversation on our [Discord server](https://discord.com/invite/feTf9x3ZSB).

If you like Gradio, please leave us a ⭐ on GitHub!

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
```