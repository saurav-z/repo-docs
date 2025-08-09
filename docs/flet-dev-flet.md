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

**Create stunning web, desktop, and mobile applications in minutes using Python and the power of Flutter.**

## Key Features

*   **Rapid Development:** Go from idea to a functional app quickly with Flet's streamlined development process.
*   **Simple Architecture:** Build stateful, multi-user, real-time Single-Page Applications (SPAs) with just Python, eliminating the need for complex frontend/backend setups.
*   **Batteries Included:**  Get started immediately with Flet's built-in web server, asset hosting, and desktop clients - no complex SDKs or dependencies required.
*   **Powered by Flutter:** Leverage the professional UI and cross-platform capabilities of Flutter, with Flet's simplified "controls" based on Flutter "widgets."
*   **Multi-Platform Deployment:** Deploy your Flet app to Windows, macOS, Linux, iOS, Android, the web (as a dynamic or static web app), or as a Progressive Web App (PWA).

## Flet App Example

Here's a basic counter app demonstrating the ease of use:

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

**To run:**

1.  **Install Flet:** `pip install 'flet[all]'`
2.  **Run your app:** `flet run counter.py`

This opens your app in a native OS window.

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
</p>

**To run as a web app:**

*   Modify your script: `ft.run(main, view=flet.AppView.WEB_BROWSER)`
*   Or use the command: `flet run --web counter.py`

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/safari.png" width="60%" />
</p>

## Get Started

*   [Flet Website](https://flet.dev)
*   [Flet Documentation](https://docs.flet.dev)
*   [Flet Roadmap](https://flet.dev/roadmap)
*   [Flet Apps Gallery](https://docs.flet.dev/gallery)

## Join the Community

*   [Discussions](https://github.com/flet-dev/flet/discussions)
*   [Discord](https://discord.gg/dzWXP8SHG8)
*   [X (Twitter)](https://twitter.com/fletdev)
*   [Bluesky](https://bsky.app/profile/fletdev.bsky.social)
*   [Email](mailto:hello@flet.dev)

## Contribute

Help improve Flet!  Read the [contribution guide](https://docs.flet.dev/contributing) and get involved.

---

**[Back to the Flet Repository](https://github.com/flet-dev/flet)**
```
Key improvements and SEO optimizations:

*   **Clear Heading Structure:** Uses `<h1>` and `<h2>` tags for better organization and readability.
*   **Concise Hook:** The opening sentence clearly and immediately states the main benefit and uses relevant keywords.
*   **Keyword Optimization:**  Uses keywords like "Python," "Flutter," "multi-platform apps," "web," "desktop," "mobile," "SPA," and "rapid development" throughout the text to improve search engine visibility.
*   **Bulleted Key Features:**  Presents the core benefits in an easy-to-scan bulleted list.
*   **Strong Call to Action:** Includes links to get started to encourage immediate engagement.
*   **Clear Instructions:**  The example includes easy-to-follow install and run instructions.
*   **Concise Summary:** The text has been trimmed and rephrased for greater clarity and impact.
*   **Back to Repo Link:** Adds a clear link back to the original repository.