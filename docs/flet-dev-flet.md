<p align="center">
  <a href="https://flet.dev"><img src="https://raw.githubusercontent.com/flet-dev/flet/refs/heads/main/media/logo/flet-logo.svg" alt="Flet logo"></a>
</p>

<p align="center">
  <em>Build stunning, multi-platform apps in Python with the power of Flutter.</em>
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

Flet is a game-changing framework that allows you to build beautiful, interactive web, desktop, and mobile applications entirely in Python, without the need for prior frontend development experience.

### Key Features

*   **Rapid Development:** Go from idea to functional app in minutes. Flet is perfect for internal tools, dashboards, prototypes, and more.
*   **Simplified Architecture:** Eliminate complex setups. Build stateful, multi-user, real-time Single-Page Applications (SPAs) with just Python.
*   **Batteries Included:** Get started quickly with your favorite IDE or text editor. No extensive SDKs or complex tooling required.
*   **Powered by Flutter:**  Leverage the power of Flutter for a polished UI that runs on any platform. Flet simplifies Flutter's model with easy-to-use controls.
*   **Multi-Platform Deployment:** Deploy your Flet apps as native desktop apps (Windows, macOS, Linux), mobile apps (iOS, Android), dynamic/static web apps, or Progressive Web Apps (PWAs).

## Getting Started with Flet

### Example: Counter App

Here's a simple "Counter" app built with Flet:

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

**To run this app:**

1.  **Install Flet:**

    ```bash
    pip install 'flet[all]'
    ```

2.  **Run the app:**

    ```bash
    flet run counter.py
    ```

    This will open the app in a native OS window.

    <p align="center">
        <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
    </p>

**To run as a web app:**

1.  Modify the last line of your script to:

    ```python
    ft.run(main, view=flet.AppView.WEB_BROWSER)
    ```

    Or use the `--web` flag:

    ```bash
    flet run --web counter.py
    ```

    <p align="center">
        <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/safari.png" width="60%" />
    </p>

## Learn More

*   [Flet Website](https://flet.dev)
*   [Flet Documentation](https://docs.flet.dev)
*   [Flet Roadmap](https://flet.dev/roadmap)
*   [Flet Apps Gallery](https://docs.flet.dev/gallery)
*   [Flet GitHub Repository](https://github.com/flet-dev/flet)

## Community

*   [Discussions](https://github.com/flet-dev/flet/discussions)
*   [Discord](https://discord.gg/dzWXP8SHG8)
*   [X (Twitter)](https://twitter.com/fletdev)
*   [Bluesky](https://bsky.app/profile/fletdev.bsky.social)
*   [Email us](mailto:hello@flet.dev)

## Contributing

Contribute to Flet's development! See the [contribution guide](https://docs.flet.dev/contributing) for details.