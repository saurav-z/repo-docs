<p align="center">
  <a href="https://flet.dev"><img src="https://raw.githubusercontent.com/flet-dev/flet/refs/heads/main/media/logo/flet-logo.svg" alt="Flet logo"></a>
</p>

<p align="center">
    <em>Build multi-platform apps in Python powered by Flutter</em>
</p>

<p align="center">
<a href="https://github.com/flet-dev/flet/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/flet-dev/flet.svg" alt="License">
</a>
<a href="https://pypi.org/project/flet" target="_blank">
    <img src="https://img.shields.io/pypi/v/flet?color=%2334D058&label=pypi" alt="Package version">
</a>
<a href="https://pepy.tech/project/flet" target="_blank">
    <img src="https://static.pepy.tech/badge/flet/month" alt="Supported Python versions">
</a>
<a href="https://pypi.org/project/flet" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/flet.svg?color=%2334D058" alt="Supported Python versions">
</a>
<a href="https://ci.appveyor.com/project/flet-dev/flet/branch/main" target="_blank">
    <img src="https://ci.appveyor.com/api/projects/status/xwablctxslvey576/branch/main?svg=true" alt="Build status">
</a>
</p>

---

## Flet: Build Beautiful, Multi-Platform Apps in Python

**Create stunning web, desktop, and mobile applications effortlessly using Python with Flet.**

### Key Features

*   **Python-First Development:** Build apps using familiar Python syntax without needing to learn complex frontend technologies.
*   **Multi-Platform Support:** Deploy your apps to web browsers, Windows, macOS, Linux, iOS, and Android from a single codebase.
*   **Flutter Powered:** Leverage the power of Flutter for beautiful, professional-looking UIs.
*   **Simplified Architecture:** Develop stateful, real-time Single-Page Applications (SPAs) with a straightforward architecture, eliminating the need for complex REST APIs.
*   **Batteries Included:** Get started quickly with a built-in web server, asset hosting, and desktop clients.
*   **Rapid Development:** Go from idea to a working app in minutes with Flet's intuitive controls and streamlined workflow.
*   **Native Performance:** Build native desktop apps, mobile apps, and web apps for great performance.

### Example: Counter App

Here's a simple counter app example built with Flet:

```python title="counter.py"
import flet as ft

def main(page: ft.Page):
    page.title = "Flet counter example"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    input = ft.TextField(value="0", text_align=ft.TextAlign.RIGHT, width=100)

    def minus_click(e):
        input.value = str(int(input.value) - 1)
        page.update()

    def plus_click(e):
        input.value = str(int(input.value) + 1)
        page.update()

    page.add(
        ft.Row(
            alignment=ft.alignment.center,
            controls=[
                ft.IconButton(ft.Icons.REMOVE, on_click=minus_click),
                input,
                ft.IconButton(ft.Icons.ADD, on_click=plus_click),
            ],
        )
    )

ft.run(main)
```

To run the app, install Flet:

```bash
pip install 'flet[all]'
```

Then run your Python script:

```bash
flet run counter.py
```

The application will open in a native OS window.

To run the same app as a web app, update the last line in your script to:

```python
ft.run(main, view=flet.AppView.WEB_BROWSER)
```

Alternatively, you can use the `--web` flag when running the `flet run` command:

```bash
flet run --web counter.py
```

### Learn More

*   [Website](https://flet.dev)
*   [Documentation](https://docs.flet.dev)
*   [Roadmap](https://flet.dev/roadmap)
*   [Apps Gallery](https://docs.flet.dev/gallery)

### Community

*   [Discussions](https://github.com/flet-dev/flet/discussions)
*   [Discord](https://discord.gg/dzWXP8SHG8)
*   [X (Twitter)](https://twitter.com/fletdev)
*   [Bluesky](https://bsky.app/profile/fletdev.bsky.social)
*   [Email us](mailto:hello@flet.dev)

### Contributing

Want to help improve Flet? Check out the [contribution guide](https://docs.flet.dev/contributing).

---

[Back to the Flet Repository](https://github.com/flet-dev/flet)
```
Key improvements and SEO optimizations:

*   **Clear Headline:**  A strong, SEO-friendly headline is the first thing users will see.
*   **One-Sentence Hook:**  Immediately grabs the user's attention and highlights the core benefit.
*   **Keyword Optimization:**  Includes relevant keywords throughout (e.g., "multi-platform apps," "Python," "Flutter," "web," "desktop," "mobile").
*   **Bulleted Key Features:**  Makes it easy to scan and understand the core advantages of Flet.
*   **Clear Structure:** Uses headings to organize the information logically, enhancing readability.
*   **Contextual Links:** Keeps original links and adds a link back to the original repo.
*   **Concise Language:**  Rephrases some text for clarity and conciseness.
*   **Stronger Call to Action (implied):** The examples encourage testing the software.
*   **Focus on Benefits:**  The features are framed in terms of the benefits they provide to the user.