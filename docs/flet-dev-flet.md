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

**Flet empowers you to create beautiful, interactive web, desktop, and mobile applications using only Python.**

### Key Features:

*   **Rapid Development:** Build apps in minutes, ideal for internal tools, prototypes, and more.
*   **Simplified Architecture:** Develop stateful, multi-user Single-Page Applications (SPAs) with Python only.
*   **Batteries Included:** Get started quickly with a built-in web server and desktop clients; no complex setups needed.
*   **Flutter Power:** Leverage the power of Flutter for professional-looking UIs across all platforms.
*   **Multi-Platform Deployment:** Deploy as desktop apps (Windows, macOS, Linux), mobile apps (iOS, Android), web apps, or PWAs.

### Flet App Example

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

1.  Install Flet:

    ```bash
    pip install 'flet[all]'
    ```
2.  Run the app:

    ```bash
    flet run counter.py
    ```

    This opens the app in a native OS window.

    <p align="center">
        <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" alt="Counter app on macOS"/>
    </p>

    To run as a web app, change the last line to:

    ```python
    ft.run(main, view=flet.AppView.WEB_BROWSER)
    ```

    Or use the `--web` flag:

    ```bash
    flet run --web counter.py
    ```

    <p align="center">
        <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/safari.png" width="60%" alt="Counter app in web browser"/>
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

*   [Contribution Guide](https://docs.flet.dev/contributing)

For more information, check out the [original repository](https://github.com/flet-dev/flet).