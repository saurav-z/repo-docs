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

## Flet: Build Stunning Multi-Platform Apps with Python and Flutter

**Flet empowers you to build beautiful, interactive web, desktop, and mobile applications entirely in Python.**  Leverage the power of Flutter without complex frontend development.

**[View the original repository](https://github.com/flet-dev/flet)**

### Key Features

*   **Rapid Development:** Create apps quickly, going from idea to a working application in minutes.
*   **Simplified Architecture:** Build stateful, multi-user, real-time Single-Page Applications (SPAs) with a Python-only codebase, eliminating the need for complex frontend and backend setups.
*   **Batteries Included:**  Develop with your favorite IDE, no complex tooling needed - Flet provides a built-in web server and desktop clients.
*   **Powered by Flutter:** Utilizes Flutter for a professional UI that delivers a consistent experience across all platforms.  Flet simplifies the Flutter model with intuitive controls.
*   **Multi-Platform Deployment:**  Deploy your Flet app as a standalone desktop app (Windows, macOS, Linux), mobile app (iOS, Android), web app, or Progressive Web App (PWA).

### Flet App Example: Counter App

Here's a simple "Counter" app example to demonstrate the ease of use.

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

**Installation:**

```bash
pip install 'flet[all]'
```

**Run the app:**

```bash
flet run counter.py
```

This launches the app in a native OS window.

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
</p>

To run as a web app:

```python
ft.run(main, view=flet.AppView.WEB_BROWSER)
```

or use the `--web` flag:

```bash
flet run --web counter.py
```

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/safari.png" width="60%" />
</p>

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

*   [Contribution guide](https://docs.flet.dev/contributing).