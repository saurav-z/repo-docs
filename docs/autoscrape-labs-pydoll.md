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

## Automate Web Tasks with Ease: Introducing Pydoll

Tired of wrestling with web drivers and complex configurations for browser automation?  **Pydoll is a Python library that makes web automation simple and effective, connecting directly to the Chrome DevTools Protocol for a natural and human-like browsing experience.**  [Visit the GitHub Repository](https://github.com/autoscrape-labs/pydoll) to get started.

### Key Features

*   **Zero Webdriver Dependency:** Eliminates compatibility headaches.
*   **Human-like Interaction Engine:** Bypass bot detection with realistic interactions.
*   **Asynchronous Performance:** Automate tasks at high speed with concurrent processing.
*   **Simplified Automation:** Focus on your logic, not the underlying complexities.
*   **Remote Control:** Control Chrome instances via WebSocket.

### What's New

*   **Remote Connections:** Control any Chrome browser remotely using WebSocket.
*   **Enhanced DOM Traversal:** `get_children_elements()` and `get_siblings_elements()` for easier navigation.
*   **WebElement Enhancements:** Improved state waiting and new public APIs for streamlined interactions.
*   **Browser-context HTTP requests:** Inherit all your browser's session state.
*   **expect_download context manager:** Robust file downloads made easy!
*   **Total browser control with custom preferences:** Customize how Chrome behaves.

### 📦 Installation

```bash
pip install pydoll-python
```

### 🚀 Getting Started

**Example: Google Search & Click**

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

**Data Extraction Example**

```python
# Extract data from a page
description = await (await tab.query(
    '//h2[contains(text(), "About")]/following-sibling::p',
    timeout=10,
)).text

# Get the rest of the data:
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

### ⚡ Advanced Features

*   **Advanced Element Search:** Find elements easily using various methods (ID, attributes, CSS selectors, XPath).
*   **Browser-context HTTP requests:** Make HTTP requests directly in the browser's JavaScript context.
*   **New expect_download() context manager:** Robust file downloads made easy.
*   **Total browser control with custom preferences:** Customize how Chrome behaves.
*   **Concurrent Automation:** Process multiple tasks simultaneously.

### 🔧 Quick Troubleshooting

*   **Browser not found?** Specify the `binary_location` in `ChromiumOptions`.
*   **Browser starts after a FailedToStartBrowser error?** Increase the `start_timeout`.
*   **Need a proxy?**  Use `--proxy-server` argument.
*   **Running in Docker?** Add `--no-sandbox` and `--disable-dev-shm-usage` arguments.

### 📚 Documentation

Explore the comprehensive [official documentation](https://pydoll.tech/) for detailed usage, examples, and API references.

### 🤝 Contributing

Contribute to Pydoll!  Review the [contribution guidelines](CONTRIBUTING.md) to get involved.

### 💖 Support My Work

Support Pydoll by [sponsoring on GitHub](https://github.com/sponsors/thalissonvs). Every contribution makes a difference!

### 💬 Spread the Word

If Pydoll helped you, give it a ⭐, share it, or tell your dev friends!

### 📄 License

Pydoll is licensed under the [MIT License](LICENSE).

<p align="center">
  <b>Pydoll</b> — Making browser automation magical!
</p>
```
Key improvements and SEO considerations:

*   **Concise and Engaging Hook:**  "Automate web tasks with ease..." grabs attention.
*   **Clear Headings:**  Uses H1, H2, and H3 for better organization and SEO structure.
*   **Keywords:** Naturally incorporates relevant keywords like "web automation," "browser automation," "Python library," "Chrome DevTools," etc.
*   **Bulleted Key Features:** Easy to scan and highlights core benefits.
*   **Action-Oriented:** Provides a clear call to action (CTA) to visit the repository.
*   **Targeted Audience:**  Speaks directly to developers by addressing common pain points.
*   **Internal Linking:** Uses `#` links for quick navigation within the README.
*   **Summarized Content:**  Condenses the original content while retaining crucial information.
*   **SEO-Friendly Formatting:** Employs Markdown formatting for readability and search engine optimization.
*   **Includes original links.**
*   **More examples added.**