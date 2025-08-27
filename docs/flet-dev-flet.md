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

**Develop stunning web, desktop, and mobile applications using Python without the complexity of frontend development with Flet.**

## Key Features

*   **Rapid Development:** Build apps quickly, going from idea to a functional application in minutes.
*   **Simple Architecture:**  Develop stateful applications in Python, without complex JavaScript frontends or REST APIs.
*   **Batteries Included:**  Get started immediately with a built-in web server, asset hosting, and desktop clients - all you need is your favorite IDE.
*   **Powered by Flutter:** Leverage the power of Flutter for a professional UI that works across all platforms.
*   **Multi-Platform Deployment:** Deploy your apps as standalone desktop apps (Windows, macOS, Linux), mobile apps (iOS, Android), web apps, or Progressive Web Apps (PWAs).

## Flet App Example: Counter

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

### How to Run

1.  **Install Flet:**

    ```bash
    pip install 'flet[all]'
    ```

2.  **Run the app:**

    ```bash
    flet run counter.py
    ```

    This will open the app in a native OS window.

    <p align="center">
        <img src="https://docs.flet-docs.pages.dev/assets/getting-started/counter-app/macos.png" width="45%" />
    </p>

3.  **Run as a Web App:**  Modify the script:

    ```python
    ft.run(main, view=flet.AppView.WEB_BROWSER)
    ```

    Or, use the `--web` flag:

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

Contribute to Flet by checking out the [contribution guide](https://docs.flet.dev/contributing).

[Back to Original Repo](https://github.com/flet-dev/flet)
```

Key improvements and SEO considerations:

*   **Clear Heading Structure:** Uses `<h1>`, `<h2>`, and bullet points for readability and SEO.
*   **Keyword Optimization:** Includes relevant keywords like "Python", "Flutter", "multi-platform apps", "web", "desktop", "mobile" throughout the content.
*   **Concise Summary:**  The one-sentence hook immediately grabs attention and summarizes Flet's core value.
*   **Feature-Rich:** Clearly lists key features using bullet points to improve readability and highlight benefits.
*   **Example Code and Instructions:** Provides a runnable example and step-by-step instructions, encouraging immediate user engagement.
*   **Call to Action:** Encourages contribution to improve engagement and SEO value.
*   **Internal Linking:** Links to other sections within the document and external resources.
*   **Back to Repo link**: Added a link at the end to the original repository.
*   **Improved Readability:** Better formatting and spacing for easier reading.
*   **SEO-Friendly Language:** Uses natural language while incorporating keywords.