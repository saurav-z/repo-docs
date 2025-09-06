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

## Flet: Build Multi-Platform Apps with Python and Flutter

**Create stunning web, desktop, and mobile applications with Python and the power of Flutter using Flet!**

Flet allows you to build cross-platform apps with Python, without the need for frontend development experience. This framework simplifies the process, allowing you to focus on building interactive and engaging applications.

### Key Features

*   **Rapid Development:**  Go from idea to a functional app in minutes, perfect for internal tools, prototypes, and more.
*   **Simplified Architecture:** Build stateful applications with Python alone; no complex JavaScript frontends or REST APIs needed.
*   **Batteries Included:**  Get started with just your favorite IDE or text editor; Flet handles the web server, asset hosting, and desktop clients.
*   **Flutter Powered:** Benefit from Flutter's professional UI capabilities, ensuring your app looks and performs great across all platforms.
*   **Multi-Platform Deployment:** Deploy your app as a standalone desktop app (Windows, macOS, Linux), mobile app (iOS, Android), web app, or PWA.

### Flet App Example: Counter App

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

**To run this example:**

1.  **Install Flet:**
    ```bash
    pip install 'flet[all]'
    ```
2.  **Run the app:**
    ```bash
    flet run counter.py
    ```

    This opens a native OS window.

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
</p>

**To run the same app as a web app:**
Update the last line in your script:

```python
ft.run(main, view=flet.AppView.WEB_BROWSER)
```

Or use the `--web` flag:

```bash
flet run --web counter.py
```

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/safari.png" width="60%" />
</p>

### Get Started

*   [Website](https://flet.dev)
*   [Documentation](https://docs.flet.dev)
*   [Apps Gallery](https://docs.flet.dev/gallery)

### Community

*   [Discussions](https://github.com/flet-dev/flet/discussions)
*   [Discord](https://discord.gg/dzWXP8SHG8)
*   [X (Twitter)](https://twitter.com/fletdev)
*   [Bluesky](https://bsky.app/profile/fletdev.bsky.social)
*   [Email us](mailto:hello@flet.dev)

### Contribute

Help improve Flet! Check out the [contribution guide](https://docs.flet.dev/contributing).

---

[Back to the original repository](https://github.com/flet-dev/flet)
```
Key improvements and SEO optimizations:

*   **Clear, concise introduction:** The opening sentence uses strong keywords and a compelling hook.
*   **Keyword Optimization:** The title and throughout the document use keywords like "Python," "Flutter," "cross-platform," "web," "desktop," and "mobile apps."
*   **Bulleted Feature List:**  Provides a scannable overview of Flet's benefits.
*   **Well-structured headings:** Improves readability and SEO.
*   **Clear Code Example:**  Provides a working example to demonstrate Flet's ease of use.
*   **Call to action:** Guides users to learn more and contribute.
*   **Concise language:** Removes unnecessary wording.
*   **Link to the Original Repo:**  Added a final link back to the original GitHub repository, as requested.