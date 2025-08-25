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

# Flet: Build Beautiful Multi-Platform Apps in Python

**Flet empowers you to create stunning web, desktop, and mobile applications using Python and the power of Flutter.**

## Key Features

*   **Rapid Development:** Quickly turn your ideas into fully functional apps in minutes.
*   **Simplified Architecture:** Build stateful, multi-user, real-time Single-Page Applications (SPAs) with Python.
*   **Batteries Included:**  Get started without complex setups – just your favorite IDE!
*   **Powered by Flutter:** Leverage Flutter's professional UI capabilities for polished app experiences.
*   **Multi-Platform Support:** Deploy your apps on any device or platform including desktop (Windows, macOS, Linux), mobile (iOS, Android), web, and PWAs.

## Flet App Example

Here's a simple "Counter" app written in Python:

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

**Installation & Running the App:**

1.  **Install Flet:**

```bash
pip install 'flet[all]'
```

2.  **Run the app as a native desktop app:**

```bash
flet run counter.py
```
<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
</p>

3.  **Run the app as a web app:**

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

[Back to top](#flet-build-beautiful-multi-platform-apps-in-python) -  [Original Repository](https://github.com/flet-dev/flet)
```

Key improvements and SEO considerations:

*   **Clear, Concise Title:**  "Flet: Build Beautiful Multi-Platform Apps in Python" is a strong, keyword-rich title.
*   **One-Sentence Hook:**  The opening sentence immediately highlights the value proposition.
*   **Keyword Optimization:**  The text uses relevant keywords like "Python," "Flutter," "multi-platform," "web," "desktop," and "mobile."
*   **Headings:**  Uses `##` for key feature headings making the content structured.
*   **Bulleted Lists:**  Key features are presented in an easy-to-scan bulleted list, improving readability.
*   **Clear Examples and Instructions:** The counter example is retained and the instructions are clear and easy to follow.
*   **Call to action:** A clear, focused call to action to start and run the app.
*   **Internal Linking:** Added a "Back to top" link for navigation and a link back to the original repository.
*   **Improved Readability:**  The overall layout is cleaner and more scannable.