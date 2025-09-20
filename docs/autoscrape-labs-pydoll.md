<p align="center">
    <img src="https://github.com/user-attachments/assets/219f2dbc-37ed-4aea-a289-ba39cdbb335d" alt="Pydoll Logo" /> <br>
</p>
<h1 align="center">Pydoll: Automate the Web, Naturally</h1>

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
</p>


## **Pydoll: The Python Library That Makes Web Automation Effortless**

Pydoll is a powerful Python library designed to simplify and streamline web automation tasks, eliminating the need for external drivers and complex configurations. It connects directly to the Chrome DevTools Protocol (CDP), making web automation intuitive and efficient. Check out the [original repository](https://github.com/autoscrape-labs/pydoll).

**Key Features:**

*   ✅ **Driverless Automation:** Directly utilizes the Chrome DevTools Protocol (CDP), eliminating the need for external drivers.
*   🎭 **Human-like Interactions:** Mimics real user behavior for more natural web interaction.
*   ⏱️ **Asynchronous Performance:** Enables high-speed automation and concurrent task execution.
*   🚀 **Easy to Use:** Install and start automating immediately with a simple, intuitive API.
*   🌐 **Remote Browser Control:** Connect and control remote Chrome instances via WebSocket.
*   💪 **Advanced Element Handling:**  Enhanced methods like `get_children_elements()` and `get_siblings_elements()` for easier DOM manipulation and `wait_until(...)` for element state management.
*   🔗 **Browser-context HTTP requests:** Make HTTP requests that automatically inherit all your browser's session state.
*   ⬇️ **Reliable File Downloads:** Simplify file download processes with the `expect_download()` context manager.
*   ⚙️ **Custom Browser Preferences:** Fine-tune browser behavior with access to hundreds of internal Chrome settings.
*   🚦 **Concurrent Automation:** Execute multiple tasks simultaneously for increased efficiency.

## **Getting Started**

### **Installation**

```bash
pip install pydoll-python
```

### **First Automation Example**

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

### **Advanced Features**

Pydoll offers a suite of advanced features for sophisticated web automation.

#### **Advanced Element Search**
Easily locate elements on the page using various methods such as find, query, and more.

```python
import asyncio
from pydoll.browser import Chrome

async def element_finding_examples():
    async with Chrome() as browser:
        tab = await browser.start()
        await tab.go_to('https://example.com')

        # Find by attributes (most intuitive)
        submit_btn = await tab.find(
            tag_name='button',
            class_name='btn-primary',
            text='Submit'
        )
        # Find by ID
        username_field = await tab.find(id='username')
        # Find multiple elements
        all_links = await tab.find(tag_name='a', find_all=True)
        # CSS selectors and XPath
        nav_menu = await tab.query('nav.main-menu')
        specific_item = await tab.query('//div[@data-testid="item-123"]')
        # With timeout and error handling
        delayed_element = await tab.find(
            class_name='dynamic-content',
            timeout=10,
            raise_exc=False  # Returns None if not found
        )
        # Advanced: Custom attributes
        custom_element = await tab.find(
            data_testid='submit-button',
            aria_label='Submit form'
        )

asyncio.run(element_finding_examples())
```

#### **Browser-context HTTP requests - game changer for hybrid automation!**
Make HTTP requests that automatically inherit all your browser's session state.

#### **New expect_download() context manager — robust file downloads made easy!**
Easily handle file downloads with the `expect_download()` context manager.

#### **Total browser control with custom preferences!**
Customize how Chrome behaves with browser preferences.

#### **Concurrent Automation**
Process multiple tasks simultaneously thanks to its asynchronous implementation.

For complete documentation, detailed examples and deep dives into all Pydoll functionalities, visit our [official documentation](https://pydoll.tech/).

## **Quick Troubleshooting**

*   **Browser not found?** Configure the `binary_location` in `ChromiumOptions`.
*   **Browser starts after a FailedToStartBrowser error?**  Increase the `start_timeout` in `ChromiumOptions`.
*   **Need a proxy?** Use the `--proxy-server` argument in `ChromiumOptions`.
*   **Running in Docker?** Add `--no-sandbox` and `--disable-dev-shm-usage` arguments in `ChromiumOptions`.

## **Contributing**

We welcome contributions!  Please refer to our [contribution guidelines](CONTRIBUTING.md) for details.

## **Support My Work**

If you find Pydoll useful, consider [supporting me on GitHub](https://github.com/sponsors/thalissonvs).

## **License**

Pydoll is licensed under the [MIT License](LICENSE).

<p align="center">
  <b>Pydoll</b> — Making browser automation magical!
</p>