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

**Flet is a Python framework that empowers developers to build stunning web, desktop, and mobile applications using the power of Flutter, without the complexities of frontend development.**  [See the original repository](https://github.com/flet-dev/flet).

## Key Features

*   **Rapid Development:**  Go from idea to a functional app in minutes, perfect for internal tools, prototypes, and more.
*   **Simplified Architecture:**  Develop complete, stateful apps in Python with a streamlined, single-codebase approach.
*   **Batteries Included:**  Start coding immediately with a built-in web server, asset hosting, and desktop clients, eliminating complex setups.
*   **Powered by Flutter:** Leverage the beauty and power of Flutter for professional-looking UIs that run flawlessly on any platform.
*   **Multi-Platform Deployment:**  Deploy your Flet apps as standalone desktop apps (Windows, macOS, Linux), mobile apps (iOS, Android), web apps, or Progressive Web Apps (PWAs).

## Flet App Example

Below is a simple "Counter" app, with a text field and two buttons to increment and decrement the counter value:

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

To run the app, install `flet`:

```bash
pip install 'flet[all]'
```

then launch the app:

```bash
flet run counter.py
```

This will open the app in a native OS window - what a nice alternative to Electron! 🙂

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
</p>

To run the same app as a web app, update the last line in your script to:

```python
ft.run(main, view=flet.AppView.WEB_BROWSER)
```

Alternatively, you can use the `--web` flag when running the `flet run` command:

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