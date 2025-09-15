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

**Flet lets you build modern, interactive web, desktop, and mobile applications in Python with ease, using the power of Flutter.**

## Key Features

*   **Python-First Development:** Build apps using Python, eliminating the need for frontend development.
*   **Multi-Platform Deployment:** Deploy your app to web, desktop (Windows, macOS, Linux), mobile (iOS, Android), or as a PWA.
*   **Powered by Flutter:** Leverage Flutter's capabilities for a professional UI that looks great on any platform.
*   **Simple Architecture:** Develop stateful, multi-user, real-time Single-Page Applications (SPAs) without complex setups.
*   **Batteries Included:** Get started quickly with built-in web server, asset hosting, and desktop clients.

## Getting Started

### Simple Counter App Example

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

**Install Flet:**

```bash
pip install 'flet[all]'
```

**Run the app:**

```bash
flet run counter.py
```

This will open the app in a native OS window.

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
</p>

**Run as a Web App:**

Update the last line to run the app in a web browser:

```python
ft.run(main, view=flet.AppView.WEB_BROWSER)
```
Or run with the `--web` flag:
```bash
flet run --web counter.py
```

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/safari.png" width="60%" />
</p>

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

Want to help improve Flet? Check out the [contribution guide](https://docs.flet.dev/contributing).

---

[Back to the Flet GitHub Repository](https://github.com/flet-dev/flet)
```

Key improvements and SEO considerations:

*   **Clear Heading Structure:** Uses `H1` for the main title and `H2` for sections, making it easy to read and understand.
*   **Keyword Optimization:**  Includes relevant keywords like "Python," "Flutter," "multi-platform," "web," "desktop," and "mobile."
*   **Concise Summary:** The one-sentence hook quickly explains what Flet is about.
*   **Bulleted Key Features:**  Highlights the core benefits, making them easy to scan.
*   **Clear Code Examples:** Keeps the code example, making it accessible for new users.
*   **Platform Highlighting:**  Clearly lists the platforms supported (web, desktop, mobile).
*   **Call to Action:** Encourages users to explore resources.
*   **Link Back to Repo:**  Provides a clear link back to the original repository.
*   **Improved Formatting:** Uses Markdown for better readability and visual appeal.
*   **Conciseness:** Removes unnecessary text while retaining all key information.