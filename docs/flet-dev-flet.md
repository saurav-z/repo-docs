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

# Flet: Build Stunning Multi-Platform Apps with Python and Flutter

**Create beautiful web, desktop, and mobile applications quickly and easily using Python with the power of Flutter.**

**[View the original repository on GitHub](https://github.com/flet-dev/flet)**

## Key Features

*   **Multi-Platform Development:** Build apps for web, desktop (Windows, macOS, Linux), and mobile (iOS, Android) from a single codebase.
*   **Python-First Approach:** Develop interactive applications using Python without needing to learn frontend technologies like HTML, CSS, or JavaScript.
*   **Powered by Flutter:** Leverage the modern UI toolkit Flutter for a professional and consistent look and feel across all platforms.
*   **Simple Architecture:**  Avoid complex architectures with Flet's stateful, monolith approach. Your app is a single-page application (SPA).
*   **Batteries Included:** Get started quickly with built-in web server, asset hosting, and desktop clients.
*   **Rapid Development:** Go from idea to a working app in minutes, ideal for internal tools, dashboards, and prototypes.
*   **Progressive Web App (PWA) Support:** Deploy your app as a PWA for enhanced web experiences.

## Flet App Example

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

### Installation and Running

1.  **Install Flet:**

    ```bash
    pip install 'flet[all]'
    ```
2.  **Run the app:**

    ```bash
    flet run counter.py
    ```

    This opens a native OS window.

    *   **Web App:**
        To run the app as a web app, change the last line in your script to:

        ```python
        ft.run(main, view=flet.AppView.WEB_BROWSER)
        ```

        Or, use the `--web` flag:

        ```bash
        flet run --web counter.py
        ```

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
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

*   Check out the [contribution guide](https://docs.flet.dev/contributing) to help improve Flet.