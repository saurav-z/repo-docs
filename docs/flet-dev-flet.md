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

**Create stunning web, desktop, and mobile applications quickly and easily using Python and the power of Flutter.**

## Key Features

*   **Develop with Python:** Build apps using your favorite language, no frontend experience needed.
*   **Cross-Platform Deployment:** Target web, desktop (Windows, macOS, Linux), iOS, Android, and PWA from a single codebase.
*   **Flutter-Powered UI:** Get a professional and modern look and feel with Flutter's UI toolkit.
*   **Simplified Architecture:** Develop stateful apps in a single Python file, simplifying your architecture compared to traditional web app development.
*   **Batteries Included:**  Flet provides a built-in web server and desktop clients, so you can start building immediately.
*   **Rapid Development:** Prototype and build interactive apps in minutes.

## Simple Architecture

Unlike traditional web development with its complex frontend, backend, and database interactions, Flet allows you to write a monolithic, stateful app in Python, resulting in a multi-user, real-time SPA.

## Built with Flutter

Flet's user interface is powered by [Flutter](https://flutter.dev/), a UI toolkit by Google. This means your applications will look and behave consistently across all platforms. Flet simplifies the Flutter model, providing ready-to-use "controls" built from smaller "widgets" with an imperative programming model.

## Platform Support

Package your Flet app as:

*   Standalone desktop app (Windows, macOS, and Linux)
*   Mobile app (iOS and Android)
*   Dynamic/static Web app
*   Progressive Web App ([PWA](https://web.dev/what-are-pwas/))

## Get Started with Flet: Counter App Example

Here's a simple "Counter" app showcasing Flet's ease of use:

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

### Installation

Install Flet using pip:

```bash
pip install 'flet[all]'
```

### Run the App

Run the `counter.py` script:

```bash
flet run counter.py
```

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

Contribute to Flet! Check out the [contribution guide](https://docs.flet.dev/contributing).

[Back to the top](#flet-build-cross-platform-apps-in-python-with-flutter) | [Original Repository](https://github.com/flet-dev/flet)
```
Key changes and improvements:

*   **SEO Optimization:** Added an SEO-friendly title, a concise introduction that is also the hook, and used relevant keywords like "cross-platform," "Python," and "Flutter."
*   **Clear Headings:** Used clear and descriptive headings (like "Key Features," "Get Started") to improve readability and organization.
*   **Bulleted Key Features:**  Highlights the core benefits and features of Flet in an easy-to-scan format.
*   **Concise Language:** Simplified language to make it more accessible and easier to understand.
*   **Call to Action:** Added a direct call to action in the "Get Started" section.
*   **Links:** Added a link back to the original repository at the end and included the original links where possible.
*   **Summarization:** Streamlined the content while retaining essential information.