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

## Flet: Build Beautiful, Multi-Platform Apps with Python and Flutter

**Flet empowers developers to create stunning web, desktop, and mobile applications using Python and the power of Flutter.**

### Key Features:

*   **Cross-Platform Development:** Build apps that run seamlessly on web, desktop (Windows, macOS, Linux), iOS, and Android.
*   **Simplified Architecture:** Develop full-stack applications with a single Python codebase - no complex JavaScript frontend or REST APIs.
*   **Rapid Development:** Quickly prototype and deploy apps with a streamlined development process.
*   **Batteries Included:** Get started without needing to manage complex SDKs, dependencies, or tooling; Flet provides a built-in web server and desktop clients.
*   **Powered by Flutter:** Leverage the UI capabilities of Flutter, ensuring a professional look and feel across all platforms.
*   **Easy-to-Use Controls:**  Flet simplifies Flutter by combining widgets into intuitive controls with an imperative programming model.

## Flet App Example

Here's a simple counter app written in Python:

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

**Running the App:**

```bash
flet run counter.py
```

This opens a native OS window. To run it as a web app:

```bash
flet run --web counter.py
```

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
</p>

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

Contribute to the project! Check out the [contribution guide](https://docs.flet.dev/contributing).

[Back to the original repo](https://github.com/flet-dev/flet)
```
Key improvements and SEO optimizations:

*   **Clear, Concise Title:**  "Flet: Build Beautiful, Multi-Platform Apps with Python and Flutter" is a descriptive and SEO-friendly title.
*   **One-Sentence Hook:** The initial sentence summarizes the core function.
*   **Keyword Rich:** Incorporated keywords like "Python," "Flutter," "cross-platform," "web," "desktop," "mobile," and "app development."
*   **Bulleted Key Features:**  Highlights the main advantages of using Flet.
*   **Simplified Language:** Replaced some of the original text with more straightforward phrasing.
*   **Clear Headings:** Uses headings (H2) for organization.
*   **Concise Example:**  The example code and instructions are kept and clarified.
*   **Call to Action:** Encourages exploration of the app.
*   **Link Back:** Adds a link back to the original repository.