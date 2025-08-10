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

## Flet: Build Amazing Multi-Platform Apps with Python and Flutter

**Flet empowers you to create stunning web, desktop, and mobile applications using only Python, eliminating the need for frontend development.**

### Key Features:

*   **Simplified Development:** Build apps quickly with a Python-only codebase, no JavaScript or complex architectures required.
*   **Multi-Platform Support:** Deploy your app on web, desktop (Windows, macOS, Linux), and mobile (iOS, Android).
*   **Powered by Flutter:** Benefit from Flutter's professional UI and cross-platform capabilities, with a simplified, Pythonic approach.
*   **Batteries Included:** Get started immediately with a built-in web server and desktop clients, without complex dependencies.
*   **Rapid Prototyping:** Ideal for internal tools, dashboards, and prototypes, allowing you to go from idea to app in minutes.
*   **PWA Support:** Easily create Progressive Web Apps for a seamless web experience.

## Flet App Example: Counter App

Here's a simple "Counter" app demonstrating the ease of building with Flet:

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

1.  Install Flet:

    ```bash
    pip install 'flet[all]'
    ```

2.  Run the app:

    ```bash
    flet run counter.py
    ```

    This will launch the app in a native OS window.

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
</p>

**To run as a web app:**

*   Modify the last line in your script:

    ```python
    ft.run(main, view=flet.AppView.WEB_BROWSER)
    ```

*   Or use the `--web` flag:

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

*   [Contribution Guide](https://docs.flet.dev/contributing)

<br>

**[Back to the Project Repository](https://github.com/flet-dev/flet)**
```

Key improvements and explanations:

*   **SEO Optimization:**  The title is now more keyword-rich ("Flet: Build Amazing Multi-Platform Apps with Python and Flutter") to attract relevant search queries.  Headings (H2 and H3) are used for structure, which helps with SEO and readability.  Keywords like "multi-platform," "Python," and "Flutter" are strategically included.
*   **Clear One-Sentence Hook:** A strong introductory sentence immediately explains what Flet does and its key benefit.
*   **Bulleted Key Features:**  The original "From idea to app in minutes" section is converted into a bulleted list of key features, making them easier to scan and understand.  Each feature is concisely described.
*   **Improved Example:** The example code is retained, but the explanation is improved and formatted for better clarity.
*   **Concise Instructions:**  The installation and running instructions are separated out from the explanatory text.
*   **Clean Formatting:**  The overall formatting is improved for better readability, with consistent use of bolding and spacing.
*   **Back to Repo Link:** A clear link is added to the original repository.
*   **Removed Redundancy:**  Some less critical details have been removed for a more focused and user-friendly README.