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

# Flet: Build Cross-Platform Apps in Python with Flutter

**Develop stunning web, desktop, and mobile applications with Python and Flutter, simplifying development and deployment across multiple platforms.**

## Key Features

*   **Cross-Platform Development:** Create apps for web, desktop (Windows, macOS, Linux), and mobile (iOS, Android) from a single codebase.
*   **Python-First Approach:** Build your entire application logic in Python, eliminating the need to learn frontend technologies like HTML, CSS, or JavaScript.
*   **Flutter Powered:** Benefit from the performance and rich UI capabilities of Flutter, delivering beautiful and responsive user interfaces.
*   **Simplified Architecture:** Design stateful, single-page applications (SPAs) with a straightforward, monolithic architecture, avoiding complex setups.
*   **Batteries Included:** Get started quickly with a built-in web server, asset hosting, and desktop client, requiring minimal setup.
*   **Rapid Development:** Go from idea to a working app in minutes, ideal for internal tools, dashboards, and prototypes.
*   **Deploy Anywhere:** Package your apps as standalone desktop applications, mobile apps, PWAs, or web apps.

## Example: Counter App

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

### Run the app:

1.  **Install Flet:**

    ```bash
    pip install 'flet[all]'
    ```

2.  **Run the app:**

    ```bash
    flet run counter.py
    ```
    (Opens in a native OS window)

    <p align="center">
        <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
    </p>

3.  **Run as a web app:**

    *   Add to your script: `ft.run(main, view=flet.AppView.WEB_BROWSER)`
    *   Or, use the command line: `flet run --web counter.py`

    <p align="center">
        <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/safari.png" width="60%" />
    </p>

## Getting Started

Ready to dive in?  Visit the [Flet Documentation](https://docs.flet.dev/) for detailed tutorials and guides.

## Resources

*   [Website](https://flet.dev)
*   [Documentation](https://docs.flet.dev)
*   [Roadmap](https://flet.dev/roadmap)
*   [Apps Gallery](https://docs.flet.dev/gallery)
*   [Flet GitHub Repository](https://github.com/flet-dev/flet)

## Community

*   [Discussions](https://github.com/flet-dev/flet/discussions)
*   [Discord](https://discord.gg/dzWXP8SHG8)
*   [X (Twitter)](https://twitter.com/fletdev)
*   [Bluesky](https://bsky.app/profile/fletdev.bsky.social)
*   [Email us](mailto:hello@flet.dev)

## Contributing

Help improve Flet! Check out the [contribution guide](https://docs.flet.dev/contributing) for details on how you can contribute.