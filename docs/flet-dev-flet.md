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

## Flet: Build Multi-Platform Apps in Python with Flutter

**Flet empowers you to create stunning web, desktop, and mobile applications using Python without any frontend development experience.**

### Key Features:

*   **Rapid Development:**  Build interactive apps quickly, from internal tools to prototypes, in minutes.
*   **Simplified Architecture:** Develop stateful, multi-user applications in a single Python codebase, eliminating complex frontend/backend setups.
*   **Batteries Included:** Get started immediately with a built-in web server, asset hosting, and desktop clients – no complex tooling required.
*   **Powered by Flutter:**  Leverage the power of Flutter for a professional UI that renders beautifully on any platform.
*   **Multi-Platform Deployment:** Deploy your app as a standalone desktop app (Windows, macOS, Linux), mobile app (iOS, Android), web app, or PWA.

### Flet App Example: Counter

Here's a simple "Counter" app example to demonstrate Flet's ease of use:

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

1.  **Install Flet:**

    ```bash
    pip install 'flet[all]'
    ```

2.  **Run the app:**

    ```bash
    flet run counter.py
    ```

    This will launch the app in a native OS window.

    <p align="center">
        <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
    </p>

    To run as a web app, modify the last line of your script or use the `--web` flag:

    ```python
    ft.run(main, view=flet.AppView.WEB_BROWSER)
    ```

    or

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

*   Want to contribute? Check out the [contribution guide](https://docs.flet.dev/contributing).

For more information, visit the [Flet GitHub repository](https://github.com/flet-dev/flet).
```

Key improvements and explanations:

*   **SEO-Optimized Heading:** Uses a clear, concise, and keyword-rich title: "Flet: Build Multi-Platform Apps in Python with Flutter".
*   **One-Sentence Hook:** Starts with a compelling sentence that immediately grabs attention and explains the core benefit:  "Flet empowers you to create stunning web, desktop, and mobile applications using Python without any frontend development experience."
*   **Key Features:** Uses bullet points for readability and to highlight the core advantages.
*   **Clear Structure:** Organizes the content with headings and subheadings for better scannability.
*   **Code Blocks Formatted:** Code examples are correctly formatted.
*   **Concise Language:** Uses clear and direct language.
*   **Call to Action:** Guides the user on how to get started.
*   **Links:**  Includes links to the documentation, the GitHub repository, and other resources.
*   **Summarized and Improved Content:**  Retains the most important information while improving its presentation.
*   **Original Repo Link at the End:** Added a final, explicit link back to the original repository for reference and discovery.