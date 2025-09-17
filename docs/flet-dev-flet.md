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

# Flet: Build Multi-Platform Apps in Python with Flutter

**Flet lets you rapidly build beautiful, multi-platform applications using only Python, powered by the Flutter framework.**

## Key Features

*   **Python-First Development:** Write your entire application logic in Python, eliminating the need for frontend development expertise.
*   **Cross-Platform Deployment:**  Build and deploy your app to Windows, macOS, Linux, iOS, Android, web browsers, and PWAs.
*   **Flutter Powered UI:** Leverage the power and polish of Flutter for a professional look and feel across all platforms.
*   **Simple Architecture:**  Flet apps are built as stateful monoliths in Python, simplifying development compared to traditional architectures.
*   **Batteries Included:** Get started quickly with a built-in web server, asset hosting, and desktop client support.
*   **Rapid Prototyping:** Ideal for internal tools, dashboards, data entry forms, and prototypes, enabling you to go from idea to app in minutes.

## How Flet Works

Flet simplifies the Flutter model by providing ready-to-use "controls" built from smaller "widgets".  This allows for an imperative programming model, making it easy to create interactive applications.

## Example: Counter App

Here's a basic "Counter" app example to illustrate how easy it is to build with Flet:

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

    This will open the app in a native OS window.

    To run as a web app:
    ```python
    ft.run(main, view=flet.AppView.WEB_BROWSER)
    ```
    Or use the `--web` flag:

    ```bash
    flet run --web counter.py
    ```

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
</p>

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/safari.png" width="60%" />
</p>


## Resources

*   **Website:** [https://flet.dev](https://flet.dev)
*   **Documentation:** [https://docs.flet.dev](https://docs.flet.dev)
*   **Roadmap:** [https://flet.dev/roadmap](https://flet.dev/roadmap)
*   **Apps Gallery:** [https://docs.flet.dev/gallery](https://docs.flet.dev/gallery)
*   **GitHub Repository:** [https://github.com/flet-dev/flet](https://github.com/flet-dev/flet)

## Get Involved

*   **Discussions:** [https://github.com/flet-dev/flet/discussions](https://github.com/flet-dev/flet/discussions)
*   **Discord:** [https://discord.gg/dzWXP8SHG8](https://discord.gg/dzWXP8SHG8)
*   **X (Twitter):** [https://twitter.com/fletdev](https://twitter.com/fletdev)
*   **Bluesky:** [https://bsky.app/profile/fletdev.bsky.social](https://bsky.app/profile/fletdev.bsky.social)
*   **Email:** [mailto:hello@flet.dev](mailto:hello@flet.dev)

## Contributing

Learn how you can help improve Flet by checking out the [contribution guide](https://docs.flet.dev/contributing).