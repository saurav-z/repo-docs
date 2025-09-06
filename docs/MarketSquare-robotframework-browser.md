# Robot Framework Browser Library: Automate Your Browser Testing with Ease

**Robot Framework Browser** is a powerful library built with Playwright, revolutionizing browser automation for Robot Framework.  [Get started with Robot Framework Browser on GitHub!](https://github.com/MarketSquare/robotframework-browser)

## Key Features:

*   **High Speed and Reliability:** Experience fast and dependable browser automation.
*   **Comprehensive Keyword Documentation:** Explore detailed documentation for all keywords.
*   **Ergonomic Selectors:**  Simplify element selection with intuitive chaining of `text`, `css`, and `xpath` selectors.
*   **JavaScript Integration:** Extend your tests with custom JavaScript functions.
*   **Asynchronous Handling:** Easily manage asynchronous HTTP requests and responses.
*   **Device Descriptor Support:**  Test across various devices with built-in device descriptors.
*   **HTTP Request and Response Handling:** Send and parse HTTP requests directly within your tests.
*   **Parallel Test Execution:**  Leverage Pabot for efficient parallel test execution.
*   **Authentication Support:** Utilize `Save Storage State` to manage authentication credentials.
*   **Robotidy Integration:** Utilize Robotidy for enhanced code formatting and maintenance.

## Installation

Follow these steps to get started:

1.  **Install Node.js:** Download and install Node.js from [https://nodejs.org/en/download/](https://nodejs.org/en/download/).  Ensure you have a supported LTS version (20 or 22).
2.  **Update pip:** Run `pip install -U pip` to ensure you have the latest version of pip.
3.  **Install the Library:** Execute `pip install robotframework-browser` from your command line.
4.  **Initialize Node Dependencies:** Run `rfbrowser init` in your shell to install necessary node dependencies and browser binaries. If `rfbrowser` is not found, try `python -m Browser.entry init`.

    *   You can skip browser binaries installation with `rfbrowser init --skip-browsers` (user is responsible for browser binary installation) or install selected browsers (chromium, firefox, or webkit) by adding them as arguments to `init` command.

## Using the Library

### Example 1: Basic Test with Robot Framework

```robotframework
*** Settings ***
Library   Browser

*** Test Cases ***
Example Test
    New Page    https://playwright.dev
    Get Text    h1    contains    Playwright
```

### Example 2: Basic Test with Python

```python
import Browser
browser = Browser.Browser()
browser.new_page("https://playwright.dev")
assert 'Playwright' in browser.get_text("h1")
browser.close_browser()
```

## Additional Resources

*   **Keyword Documentation:** [https://marketsquare.github.io/robotframework-browser/Browser.html](https://marketsquare.github.io/robotframework-browser/Browser.html)
*   **Web Page:** [https://robotframework-browser.org/](https://robotframework-browser.org/)
*   **Docker Images:** [https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable](https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable)  (See [docker/README.md](https://github.com/MarketSquare/robotframework-browser/blob/main/docker/README.md))
*   **Example Extensions:** [https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015](https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015)
*   **Robot Framework Browser Extensions**: [https://github.com/MarketSquare/robotframework-browser-extensions](https://github.com/MarketSquare/robotframework-browser-extensions)

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.

## Core Team

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

[See the full list of contributors](https://github.com/MarketSquare/robotframework-browser#contributors-) for their invaluable contributions.
Supported by [Robocorp](https://robocorp.com/) through [Robot Framework Foundation](https://robotframework.org/foundation/).