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
  • 🔗  <a href="https://github.com/autoscrape-labs/pydoll">View on GitHub</a>
</p>

## Automate Web Tasks with Ease Using Pydoll

Pydoll is a powerful Python library that simplifies web automation by connecting directly to the Chrome DevTools Protocol (CDP).  This eliminates the need for external drivers, and simplifies complex configurations.  Focus on your automation logic, not the underlying complexity.

### Key Features:

*   **Zero WebDriver Dependencies:** Avoid driver compatibility headaches.
*   **Human-Like Interaction Engine:** Bypass CAPTCHAs like reCAPTCHA v3 and Turnstile (depending on reputation).
*   **Asynchronous Performance:** Execute tasks concurrently for speed.
*   **Intuitive Interactions:** Mimic real user behavior for effective automation.
*   **Simplified Setup:**  Install and start automating instantly.
*   **Remote Browser Control:** Connect to and control remote Chrome instances via WebSocket.
*   **DOM Navigation Helpers:** `get_children_elements()` and `get_siblings_elements()` simplify complex layout traversal.
*   **Enhanced WebElement API:** `wait_until()`, `is_visible()`, `is_interactable()`, `is_on_top()`, and `execute_script()` for robust element handling.
*   **Browser-context HTTP requests:** Make HTTP requests that automatically inherit all your browser's session state.
*   **expect_download context manager:** Robust file downloads made easy.
*   **Total browser control with custom preferences:** Customize how Chrome behaves.
*   **Concurrent Automation:** Run multiple automation tasks at the same time!

### What's New:

*   **Remote connections via WebSocket:** Control any Chrome from anywhere!
*   **Navigate the DOM like a pro: get_children_elements() and get_siblings_elements()**
*   **WebElement: state waiting and new public APIs**

### Installation

```bash
pip install pydoll-python
```

### Getting Started: Automating a Google Search

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

This example demonstrates how easy it is to automate a Google search and click on a result using Pydoll.

### Extracting Data from a Page

This example shows how to extract the project description, number of stars, forks, issues, and pull requests from a GitHub page:

```python
description = await (await tab.query(
    '//h2[contains(text(), "About")]/following-sibling::p',
    timeout=10,
)).text

number_of_stars = await (await tab.find(
    id='repo-stars-counter-star'
)).text

number_of_forks = await (await tab.find(
    id='repo-network-counter'
)).text
number_of_issues = await (await tab.find(
    id='issues-repo-tab-count',
)).text
number_of_pull_requests = await (await tab.find(
    id='pull-requests-repo-tab-count',
)).text

data = {
    'description': description,
    'number_of_stars': number_of_stars,
    'number_of_forks': number_of_forks,
    'number_of_issues': number_of_issues,
    'number_of_pull_requests': number_of_pull_requests,
}
print(data)
```

### Custom Configurations

Configure your browser with custom settings, such as using a proxy, window size, and custom Chrome binary location:

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

### Advanced Features

Explore advanced element search, browser-context HTTP requests, downloading files, concurrent automation, and many more features to enhance your web automation capabilities.

*   **Advanced Element Search:** Powerful methods for finding elements (find, query).
*   **Browser-context HTTP requests:** Make HTTP requests that automatically inherit all your browser's session state.
*   **expect_download context manager:** Robust file downloads made easy.
*   **Total browser control with custom preferences!** Customize how Chrome behaves.
*   **Concurrent Automation:** Run multiple automation tasks at the same time!

See the examples in the original README, and in the documentation.

### Quick Troubleshooting

Address common issues like browser not found, slow startup, proxies, and Docker setups.

### Documentation

Refer to the [official documentation](https://pydoll.tech/) for detailed guides, API references, and advanced techniques.

### Contributing

Help us improve Pydoll! Review the [contribution guidelines](CONTRIBUTING.md) to get started.

### Support My Work

Show your appreciation by [sponsoring me on GitHub](https://github.com/sponsors/thalissonvs) or supporting the project through starring and sharing.

### License

Pydoll is licensed under the [MIT License](LICENSE).