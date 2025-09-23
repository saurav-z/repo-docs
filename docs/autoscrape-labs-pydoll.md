<p align="center">
    <img src="https://github.com/user-attachments/assets/219f2dbc-37ed-4aea-a289-ba39cdbb335d" alt="Pydoll Logo" /> <br>
    <a href="https://github.com/autoscrape-labs/pydoll">
        <img src="https://img.shields.io/github/stars/autoscrape-labs/pydoll?style=social" alt="GitHub stars">
    </a>
</p>

<h1 align="center">Pydoll: Effortless Web Automation with Python</h1>

Pydoll empowers you to automate web interactions seamlessly, eliminating the complexities of traditional web drivers.

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

## Key Features

*   **Driverless Automation:** Bypasses webdriver compatibility issues by connecting directly to the Chrome DevTools Protocol (CDP).
*   **Human-Like Interaction Engine:** Mimics realistic user behavior for effective automation, including bypassing anti-bot measures.
*   **Asynchronous Performance:** Enables high-speed automation and the ability to manage multiple tasks concurrently.
*   **Simplified Setup:** Installs easily, letting you quickly start automating web tasks.

## What's New
*(Summarized from original README)*

### Remote Connections via WebSocket
Connect to and control remote/CI browsers via WebSocket, facilitating automation in various environments.

### DOM Navigation Helpers
New methods `get_children_elements()` and `get_siblings_elements()` to simplify DOM traversal and streamline scraping logic.

### WebElement Enhancements
Added public methods (`is_visible()`, `is_interactable()`, `is_on_top()`, and `execute_script()`) to improve element state validation. Also added `wait_until()` on `WebElement` to await element states.

## 📦 Installation

```bash
pip install pydoll-python
```

## 🚀 Getting Started

*(Summarized from original README)*

Quickly automate web tasks with examples that show how to perform a Google search and extract data using xpath queries, customizable configurations.

## ⚡ Advanced Features

### Advanced Element Search
Offers multiple methods to find elements on a page. Search options include the intuitive `find` method with attributes and the versatile `query` method with CSS selectors and XPath queries.

### Browser-context HTTP requests
The `tab.request` property gives you a `requests`-like interface that executes HTTP calls directly in the browser's JavaScript context. This means every request automatically gets cookies, authentication headers, CORS policies, and session state.

### New expect_download() context manager
The `tab.expect_download()` context manager offers a simplified way to handle file downloads.

### Total browser control with custom preferences
The new `browser_preferences` system gives you access to hundreds of internal Chrome settings.

### Concurrent Automation
Pydoll's asynchronous nature allows for simultaneous processing of multiple tasks.

## 🔧 Quick Troubleshooting

*(Summarized from original README)*

Provides solutions for common issues: Browser not found, FailedToStartBrowser errors, proxy usage, and running in Docker.

## 📚 Documentation

For complete documentation, detailed examples and deep dives into all Pydoll functionalities, visit our [official documentation](https://pydoll.tech/).

## 🤝 Contributing

Contribute to the project by following the [contribution guidelines](CONTRIBUTING.md) and best practices to enhance Pydoll.

## 💖 Support My Work

Show your support by sponsoring the project on GitHub ([https://github.com/sponsors/thalissonvs](https://github.com/sponsors/thalissonvs)) or by other means.

## 💬 Spread the word

Share the project and support it with your feedback.

## 📄 License

Pydoll is licensed under the [MIT License](LICENSE).

<p align="center">
  <b>Pydoll</b> — Making browser automation magical!
</p>

[Back to the original repo](https://github.com/autoscrape-labs/pydoll)