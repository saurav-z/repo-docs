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

## Flet: Build Cross-Platform Apps with Python and Flutter

**Flet empowers you to rapidly build stunning, multi-platform applications using only Python.**

### Key Features:

*   **Simplified Development:** Build apps without complex frontend technologies - just Python.
*   **Multi-Platform Support:** Deploy your apps to web, desktop (Windows, macOS, Linux), and mobile (iOS, Android).
*   **Powered by Flutter:** Leverage the power of Flutter for beautiful, professional-looking UIs.
*   **Batteries Included:** Get started quickly with a built-in web server, asset hosting, and desktop clients - no complex setup required.
*   **Rapid Prototyping:** Quickly go from idea to a functional app in minutes.
*   **Simple Architecture:** Develop stateful, multi-user apps with a monolithic Python codebase, no separate backend or API needed.

### Example: Counter App

Here's a simple counter app built with Flet:

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

Then, run the script:

```bash
flet run counter.py
```

This opens the app as a native OS window.

To run as a web app, change the last line:

```python
ft.run(main, view=flet.AppView.WEB_BROWSER)
```

or use the `--web` flag:

```bash
flet run --web counter.py
```

### Learn More

*   [Website](https://flet.dev)
*   [Documentation](https://docs.flet.dev)
*   [Roadmap](https://flet.dev/roadmap)
*   [Apps Gallery](https://docs.flet.dev/gallery)
*   **[Flet GitHub Repository](https://github.com/flet-dev/flet)**

### Community

*   [Discussions](https://github.com/flet-dev/flet/discussions)
*   [Discord](https://discord.gg/dzWXP8SHG8)
*   [X (Twitter)](https://twitter.com/fletdev)
*   [Bluesky](https://bsky.app/profile/fletdev.bsky.social)
*   [Email us](mailto:hello@flet.dev)

### Contributing

*   [Contribution Guide](https://docs.flet.dev/contributing)
```
Key improvements:

*   **SEO Optimization:**  Includes keywords like "cross-platform," "Python," and "Flutter" in the headings and descriptions.
*   **One-Sentence Hook:**  Starts with a compelling sentence to grab the reader's attention.
*   **Clear Headings:** Uses clear, concise headings for each section.
*   **Bulleted Key Features:**  Presents key features in an easy-to-read bulleted list.
*   **Concise Summary:**  Provides a more focused summary of the framework's capabilities.
*   **GitHub Link:** Added a link back to the original repository for easy access.
*   **Reorganized Content:** Improved the flow and readability of the original content.