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

# Flet: Build Cross-Platform Apps with Python and Flutter

**Create beautiful, multi-platform applications in minutes using Python and the power of Flutter.**

Flet is a revolutionary framework that allows you to build web, desktop, and mobile applications with Python, eliminating the need for complex frontend technologies. With Flet, you can rapidly prototype and deploy interactive applications.

## Key Features:

*   ⚡️ **Rapid Development:**  Go from idea to functional app in minutes with a simple Python-based architecture.
*   📐 **Simplified Architecture:**  Develop stateful apps using Python without the complexity of JavaScript frontends or REST APIs.  Flet handles the complexities, providing a real-time, multi-user SPA (Single-Page Application).
*   🔋 **Batteries Included:**  Start developing immediately with your favorite IDE. Flet includes a built-in web server, asset hosting, and desktop clients, reducing dependencies.
*   <img src="https://storage.googleapis.com/cms-storage-bucket/icon_flutter.4fd5520fe28ebf839174.svg" width="18" style="vertical-align: middle;" /> **Powered by Flutter:** Benefit from Flutter's professional UI and cross-platform capabilities.  Flet simplifies Flutter's model, combining widgets into easy-to-use controls with an imperative programming model.
*   📱 **Multi-Platform Deployment:**  Deploy your Flet apps as standalone desktop apps (Windows, macOS, Linux), mobile apps (iOS, Android), web apps, or PWAs (Progressive Web Apps).

## Flet App Example: Counter App

This simple counter app demonstrates how easy it is to get started with Flet.

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

Then, run your app:

```bash
flet run counter.py
```

This will launch the app in a native OS window.

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
</p>

To run as a web app, modify the last line:

```python
ft.run(main, view=flet.AppView.WEB_BROWSER)
```

Or, use the `--web` flag:

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
See the original repo on GitHub: [flet-dev/flet](https://github.com/flet-dev/flet)