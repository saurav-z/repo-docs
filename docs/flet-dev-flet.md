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

**Create stunning web, desktop, and mobile applications quickly and easily using Python with Flet.**

## Key Features

*   **Rapid Development:** Build apps in minutes with an intuitive, Python-based framework.
*   **Simple Architecture:** Develop stateful, multi-user, real-time Single-Page Applications (SPAs) without complex backend setups.
*   **Batteries Included:** Get started immediately with a built-in web server and desktop clients; no extensive SDKs or dependencies required.
*   **Powered by Flutter:** Benefit from a professional UI built with Flutter, ensuring a consistent look and feel across all platforms.
*   **Multi-Platform Deployment:** Deploy your app as a desktop app (Windows, macOS, Linux), mobile app (iOS, Android), web app, or PWA.

## How Flet Works

Flet simplifies app development by using an imperative programming model that combines smaller "widgets" into ready-to-use "controls." With Flet, you can create interactive apps using just Python, leveraging the power of Flutter for a polished, cross-platform user interface.

## Flet App Example: Counter App

Here's a simple "Counter" app to demonstrate Flet's ease of use:

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

### Run the App

1.  **Install Flet:**

    ```bash
    pip install 'flet[all]'
    ```
2.  **Run the app (desktop):**

    ```bash
    flet run counter.py
    ```
    *(This opens the app in a native OS window.)*

    <p align="center">
        <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
    </p>

3.  **Run the app (web):**
    *   Change the last line in your script to:
        ```python
        ft.run(main, view=flet.AppView.WEB_BROWSER)
        ```
        *Or use the `--web` flag when running the `flet run` command:*
        ```bash
        flet run --web counter.py
        ```

    <p align="center">
        <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/safari.png" width="60%" />
    </p>

## Learn More

*   [Official Website](https://flet.dev)
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

Help improve Flet! Check out the [contribution guide](https://docs.flet.dev/contributing).

---

**[Visit the original Flet repository on GitHub](https://github.com/flet-dev/flet)**