<p align="center">
    <img src="https://github.com/user-attachments/assets/219f2dbc-37ed-4aea-a289-ba39cdbb335d" alt="Pydoll Logo" /> <br>
</p>

<h1 align="center">Pydoll: Automate the Web, Effortlessly</h1>

<p align="center">
    <a href="https://github.com/autoscrape-labs/pydoll/stargazers"><img src="https://img.shields.io/github/stars/autoscrape-labs/pydoll?style=social"></a>
    <a href="https://codecov.io/gh/autoscrape-labs/pydoll" >
        <img src="https://codecov.io/gh/autoscrape-labs/pydoll/graph/badge.svg?token=40I938OGM9"/>
    </a>
    <img src="https://github.com/autoscrape-labs/pydoll/actions/workflows/tests.yml/badge.svg" alt="Tests">
    <img src="https://github.com/autoscrape-labs/pydoll/actions/workflows/ruff-ci.yml/badge.svg" alt="Ruff CI">
    <img src="https://github.com/autoscrape-labs/pydoll/actions/workflows/mypy.yml/badge.svg" alt="MyPy CI">
    <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="Python >= 3.10">
    <a href="https://deepwiki.com/autoscrape-labs/pydoll"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>

<p align="center">
  📖 <a href="https://pydoll.tech/">Documentation</a> •
  🚀 <a href="#-getting-started">Getting Started</a> •
  ⚡ <a href="#-advanced-features">Advanced Features</a> •
  🤝 <a href="#-contributing">Contributing</a> •
  💖 <a href="#-support-my-work">Support My Work</a>
  <br>
  <a href="https://github.com/autoscrape-labs/pydoll">
  <img src="https://img.shields.io/badge/View%20on%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="View on GitHub"></a>
</p>

**Pydoll simplifies web automation by connecting directly to the Chrome DevTools Protocol (CDP), offering a robust, user-friendly experience without the need for external drivers.**

## Key Features

*   **Driverless Automation:** Eliminate webdriver compatibility issues.
*   **Human-like Interactions:** Bypass bot detection with realistic user behavior.
*   **Asynchronous Performance:** Execute tasks concurrently for speed and efficiency.
*   **Simplified Implementation:** Install and start automating immediately.
*   **Remote Browser Control:** Connect to and control remote Chrome instances via WebSocket.
*   **Advanced DOM Traversal:** Utilize `get_children_elements()` and `get_siblings_elements()` for efficient DOM manipulation.
*   **Enhanced WebElement Control:** Employ `wait_until()`, `is_visible()`, `is_interactable()`, `is_on_top()`, and `execute_script()` for reliable element interaction.
*   **Browser-Context HTTP Requests:** Make requests with inherited browser session, cookies, and CORS policies.
*   **Robust File Downloads:** Employ `expect_download()` context manager for reliable and efficient file handling.
*   **Custom Browser Preferences:** Total control over Chrome behavior with hundreds of internal settings.
*   **Concurrent Automation:** Process multiple tasks simultaneously with asynchronous implementation.

## What's New

### Remote Connections via WebSocket

Control any Chrome instance remotely, perfect for CI/CD and debugging.

```python
from pydoll.browser.chromium import Chrome

chrome = Chrome()
tab = await chrome.connect('ws://YOUR_HOST:9222/devtools/browser/XXXX')

# Full power unlocked: navigation, element automation, requests, events…
await tab.go_to('https://example.com')
title = await tab.execute_script('return document.title')
print(title)
```

### DOM Traversal Helpers: get_children_elements() and get_siblings_elements()

Simplify navigation through complex layouts.

```python
# Grab direct children of a container
container = await tab.find(id='cards')
cards = await container.get_children_elements(max_depth=1)

# Walk horizontal lists without re-querying the DOM
active = await tab.find(class_name='item-active')
siblings = await active.get_siblings_elements()
```

### WebElement State Waiting and New Public APIs

Improve reliability with `wait_until()` and expanded `WebElement` methods.

```python
# Wait until it becomes visible OR the timeout expires
await element.wait_until(is_visible=True, timeout=5)

# Visually outline the element via JS
await element.execute_script("this.style.outline='2px solid #22d3ee'")
```

### Advanced Features

Pydoll Offers a series of advanced features to please even the most demanding users, including:

*   **Advanced Element Search**
*   **Browser-context HTTP requests**
*   **New expect_download() context manager**
*   **Total browser control with custom preferences**
*   **Concurrent Automation**

## 📦 Installation

```bash
pip install pydoll-python
```

## 🚀 Getting Started

### Your first automation
```python
import asyncio

from pydoll.browser import Chrome
from pydoll.constants import Key

async def google_search(query: str):
    async with Chrome() as browser:
        tab = await browser.start()
        await tab.go_to('https://www.google.com')
        search_box = await tab.find(tag_name='textarea', name='q')
        await search_box.insert_text(query)
        await search_box.press_keyboard_key(Key.ENTER)
        await (await tab.find(
            tag_name='h3',
            text='autoscrape-labs/pydoll',
            timeout=10,
        )).click()
        await tab.find(id='repository-container-header', timeout=10)

asyncio.run(google_search('pydoll python'))
```
### Custom Configurations
```python
from pydoll.browser import Chrome
from pydoll.browser.options import ChromiumOptions as Options

async def custom_automation():
    # Configure browser options
    options = Options()
    options.add_argument('--proxy-server=username:password@ip:port')
    options.add_argument('--window-size=1920,1080')
    options.binary_location = '/path/to/your/browser'
    options.start_timeout = 20

    async with Chrome(options=options) as browser:
        tab = await browser.start()
        # Your automation code here
        await tab.go_to('https://example.com')
        # The browser is now using your custom settings

asyncio.run(custom_automation())
```

## 🔧 Quick Troubleshooting

*   **Browser not found?** Configure `binary_location` in `ChromiumOptions`.
*   **Browser starts after a FailedToStartBrowser error?** Increase `start_timeout` in `ChromiumOptions`.
*   **Need a proxy?**  Add `--proxy-server=your-proxy:port` to arguments.
*   **Running in Docker?**  Add `--no-sandbox` and `--disable-dev-shm-usage` to arguments.

## 📚 Documentation

Comprehensive documentation, examples, and API references are available on our [official documentation](https://pydoll.tech/).

## 🤝 Contributing

Contribute to Pydoll!  See our [contribution guidelines](CONTRIBUTING.md).

## 💖 Support

Support Pydoll on GitHub!  Consider becoming a [sponsor](https://github.com/sponsors/thalissonvs).

## 📄 License

Pydoll is licensed under the [MIT License](LICENSE).