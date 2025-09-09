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

**Create stunning web, desktop, and mobile applications in Python quickly and easily with Flet.**

Flet is a powerful framework that empowers Python developers to build user interfaces for web, desktop, and mobile applications without any prior frontend experience.

### Key Features:

*   ⚡️ **Rapid Development:** Go from idea to a working app in minutes, ideal for internal tools, prototypes, and more.
*   📐 **Simplified Architecture:** Develop stateful, multi-user, real-time Single-Page Applications (SPAs) with Python only, eliminating complex frontend/backend separation.
*   🔋 **Batteries Included:** Start coding immediately with Flet, no need for complex SDKs or dependencies. It includes a built-in web server and desktop clients.
*   📱 **Powered by Flutter:** Build professional-looking UIs with Flutter, ensuring a consistent experience across all platforms. Flet simplifies Flutter's model with easy-to-use controls.
*   🌍 **Multi-Platform Deployment:** Package your app for Windows, macOS, Linux, iOS, Android, web apps (dynamic/static), and Progressive Web Apps (PWAs).

## Flet App Example: Counter

Here's a simple "Counter" app example:

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

To run the app:

1.  Install Flet: `pip install 'flet[all]'`
2.  Run the app: `flet run counter.py`

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
</p>

To run as a web app:

*   Modify the last line: `ft.run(main, view=flet.AppView.WEB_BROWSER)`
*   Or, use the `--web` flag: `flet run --web counter.py`

<p align="center">
    <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/safari.png" width="60%" />
</p>

## Get Started

*   **[Flet Website](https://flet.dev)**
*   **[Flet Documentation](https://docs.flet.dev)**
*   **[Flet Roadmap](https://flet.dev/roadmap)**
*   **[Flet Apps Gallery](https://docs.flet.dev/gallery)**

## Community

*   [Discussions](https://github.com/flet-dev/flet/discussions)
*   [Discord](https://discord.gg/dzWXP8SHG8)
*   [X (Twitter)](https://twitter.com/fletdev)
*   [Bluesky](https://bsky.app/profile/fletdev.bsky.social)
*   [Email us](mailto:hello@flet.dev)

## Contributing

Contribute to Flet's development: [Contribution Guide](https://docs.flet.dev/contributing).

**[Visit the Flet Repository on GitHub](https://github.com/flet-dev/flet)**