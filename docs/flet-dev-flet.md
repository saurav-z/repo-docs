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

# Flet: Build Multi-Platform Apps with Python and Flutter

**Flet empowers you to create beautiful, multi-platform applications for web, desktop, and mobile using only Python, eliminating the need for frontend development.** Get started by exploring the [original repository](https://github.com/flet-dev/flet).

## Key Features

*   **Python-First Development:** Build apps using Python without needing to learn JavaScript or other frontend technologies.
*   **Multi-Platform Support:** Deploy your app on web, desktop (Windows, macOS, Linux), and mobile (iOS, Android) platforms.
*   **Powered by Flutter:** Leverage the power and flexibility of Flutter for a professional and modern UI, with Flet simplifying the development model.
*   **Rapid Development:** Quickly prototype and build interactive apps with a streamlined development process.
*   **Simple Architecture:**  Develop stateful apps in Python with a built-in web server and client support.
*   **Batteries Included:**  No complex SDKs or tooling needed; start coding immediately using your favorite IDE.

## Flet App Example

Here's a simple counter app to demonstrate Flet's ease of use:

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

To run the app, install Flet:

```bash
pip install 'flet[all]'
```

And then run your Python script:

```bash
flet run counter.py
```

You can also run the app as a web app by updating the last line of your script:

```python
ft.run(main, view=flet.AppView.WEB_BROWSER)
```

Or by using the `--web` flag:

```bash
flet run --web counter.py
```

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

*   [Contribution guide](https://docs.flet.dev/contributing)