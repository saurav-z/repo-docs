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

# Flet: Build Beautiful, Multi-Platform Apps in Python with Flutter

Flet is a revolutionary framework empowering Python developers to craft stunning web, desktop, and mobile applications without complex frontend code.  ([See the original repository](https://github.com/flet-dev/flet))

## Key Features

*   **Rapid Development:**  Go from idea to a fully functional app in minutes, perfect for internal tools, prototypes, and more.
*   **Simplified Architecture:** Build stateful, multi-user, real-time Single-Page Applications (SPAs) with a Python-only codebase. No need for complex JavaScript, REST APIs, or database setups.
*   **Batteries Included:**  Get started immediately with your favorite IDE or text editor.  Flet includes a built-in web server and desktop clients, eliminating the need for extensive SDKs or dependencies.
*   **Powered by Flutter:** Leverage the power of Flutter for a polished, professional UI that renders beautifully on any platform.
*   **Multi-Platform Deployment:** Deploy your app as a standalone desktop application (Windows, macOS, Linux), mobile app (iOS, Android), or as a web app (dynamic/static) or Progressive Web App (PWA).

## Flet App Example

Here's a simple "Counter" app demonstrating Flet's ease of use:

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

**To run the app:**

1.  **Install Flet:**

```bash
pip install 'flet[all]'
```

2.  **Run the app:**

```bash
flet run counter.py
```

This will launch a native OS window.  To run as a web app:

```python
ft.run(main, view=flet.AppView.WEB_BROWSER)
```

or use the `--web` flag:

```bash
flet run --web counter.py
```

## Learn More

*   [Website](https://flet.dev)
*   [Documentation](https://docs.flet.dev)
*   [Roadmap](https://flet.dev/roadmap)
*   [Apps Gallery](https://docs.flet.dev/gallery)

## Community

*   [Discussions](https://github.com/flet-dev/flet/discussions)
*   [Discord](https://discord.gg/dzWXP8SHG8)
*   [X (Twitter)](https://twitter.com/fletdev)
*   [Bluesky](https://bsky.app/profile/fletdev.bsky.social)
*   [Email us](mailto:hello@flet.dev)

## Contributing

Help make Flet even better!  Check out the [contribution guide](https://docs.flet.dev/contributing).
```
Key improvements:

*   **SEO Optimization:**  Added a clear, keyword-rich title ("Flet: Build Beautiful, Multi-Platform Apps in Python with Flutter") and a compelling one-sentence hook.
*   **Clear Headings:** Organized the README with descriptive headings (Key Features, Flet App Example, Learn More, Community, Contributing).
*   **Bulleted Features:** Uses bullet points for readability and emphasizes key advantages.
*   **Concise Summarization:** Condensed the original text while retaining essential information.
*   **Focus on Benefits:** Highlights what users *get* from Flet (rapid development, simplified architecture, multi-platform deployment).
*   **Link to Original Repo:** Includes a prominent link back to the original GitHub repository.
*   **Improved Code Formatting:** Improved the formatting of code blocks.