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

# Flet: Build Beautiful, Multi-Platform Apps with Python and Flutter

**Create stunning web, desktop, and mobile applications quickly and easily using Python and the power of Flutter.**

## Key Features

*   **Rapid Development:**  Build apps from idea to execution in minutes, perfect for internal tools, dashboards, prototypes, and more.
*   **Simplified Architecture:**  Develop full-stack applications in a single Python file, eliminating complex frontend/backend setups.
*   **Batteries Included:**  Get started immediately with no SDKs or complex dependencies; Flet provides a built-in web server and desktop clients.
*   **Flutter Powered:**  Leverage Flutter's professional UI capabilities for beautiful and consistent apps across all platforms.
*   **Cross-Platform Deployment:**  Deploy your app as a standalone desktop application (Windows, macOS, Linux), mobile app (iOS, Android), web app, or Progressive Web App (PWA).

## Getting Started

### Example: Counter App

Here's a simple "Counter" app example in Python:

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

1.  **Install Flet:**

    ```bash
    pip install 'flet[all]'
    ```

2.  **Run the app (Desktop):**

    ```bash
    flet run counter.py
    ```

    <p align="center">
        <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
    </p>

3.  **Run the app (Web):**

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

Interested in contributing to Flet?  Check out the [contribution guide](https://docs.flet.dev/contributing).

---

**[Back to the Flet Repository](https://github.com/flet-dev/flet)**
```
Key improvements and explanations:

*   **Clear, concise hook:** The one-sentence hook immediately grabs attention and highlights the core value proposition (building multi-platform apps with Python and Flutter).
*   **SEO-friendly Heading:** Changed the title to be more descriptive, and added subheadings.
*   **Bulleted Key Features:**  Improved readability and clarity of the key selling points.
*   **Concise Language:** Streamlined wording for brevity and impact.
*   **Improved Code Example:**  Maintained the example but formatted it for better readability.
*   **Clear Installation Instructions:**  Simplified the installation and running instructions.
*   **Direct Link Back:** Includes a clear link back to the original GitHub repository for easy navigation and context.
*   **Structure:** Uses headings and spacing to improve the organization and readability of the information.
*   **Keywords:** The text now includes keywords like "Python," "Flutter," "multi-platform," "web," "desktop," and "mobile" to boost search engine visibility.