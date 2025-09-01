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

## Flet: Build Cross-Platform Apps with Python and Flutter

**Create stunning web, desktop, and mobile applications in minutes using Python and the power of Flutter.**

### Key Features

*   **Rapid Development:** Quickly build interactive apps with a simple, Python-based framework.
*   **Simplified Architecture:** Develop stateful, multi-user, real-time Single-Page Applications (SPAs) without complex backend setups.
*   **Batteries Included:**  Get started easily with a built-in web server, asset hosting, and desktop clients, requiring minimal dependencies.
*   **Powered by Flutter:**  Leverage Flutter's capabilities to create professional-looking UI's, ensuring consistent performance across all platforms.
*   **Multi-Platform Deployment:** Deploy your app as a standalone desktop application (Windows, macOS, Linux), mobile app (iOS, Android), dynamic/static web app, or a Progressive Web App (PWA).

### Flet App Example

Here's a basic counter app to illustrate Flet's ease of use:

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

To run the app, install `flet`:

```bash
pip install 'flet[all]'
```

Then launch the app:

```bash
flet run counter.py
```

This will open the app in a native OS window.

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
</p>

To run the same app as a web app, update the last line in your script to:

```python
ft.run(main, view=flet.AppView.WEB_BROWSER)
```

Alternatively, you can use the `--web` flag when running the `flet run` command:

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

Interested in improving Flet? Explore the [contribution guide](https://docs.flet.dev/contributing).

---
**[Back to the original repository](https://github.com/flet-dev/flet)**
```

Key improvements and explanations:

*   **SEO Optimization:**  Includes relevant keywords like "cross-platform apps," "Python," and "Flutter" in the title and throughout the text.
*   **One-Sentence Hook:** Starts with a strong introductory sentence to immediately grab the reader's attention.
*   **Clear Headings:** Uses headings (H2 and H3) to organize content and improve readability.
*   **Bulleted Key Features:** Highlights key benefits in a concise and easily scannable format.
*   **Concise Summarization:**  Rephrases information to be more direct and easier to digest.
*   **Call to Action (Learn More):** Encourages the reader to explore further resources.
*   **Clear Examples:** The example code is retained as it's a core part of the documentation.
*   **Added link back to original repo** This is crucial for directing people to the source code.