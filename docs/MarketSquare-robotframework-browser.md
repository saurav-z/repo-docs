# Robot Framework Browser Library: Automate Web Testing with Ease

[Robot Framework Browser](https://github.com/MarketSquare/robotframework-browser) is a powerful and reliable browser automation library built on [Playwright](https://playwright.dev) for [Robot Framework](https://robotframework.org), designed to bring speed, reliability, and visibility to your web testing.

## Key Features

*   **Fast & Reliable:** Leveraging Playwright for high-speed and dependable browser automation.
*   **Ergonomic Selectors:**  Chain `text`, `css`, and `xpath` selectors for intuitive element selection.
*   **JavaScript Integration:** Easily extend functionality with JavaScript extensions.
*   **Asynchronous HTTP Handling:**  Asynchronously wait for HTTP requests and responses.
*   **Device Descriptors:**  Test responsive designs with device emulation.
*   **HTTP Request Support:**  Send and parse HTTP requests directly.
*   **Parallel Test Execution:** Utilize Pabot for efficient parallel test execution.

## Installation

### Prerequisites:

*   Python 3.9 or newer
*   Node.js (LTS versions 20, 22, and 24 recommended)

### Steps:

1.  **Install Node.js:**  Download and install from [https://nodejs.org/en/download/](https://nodejs.org/en/download/)
2.  **Update pip:** `pip install -U pip`
3.  **Install the library:** `pip install robotframework-browser`
4.  **Initialize Node dependencies:**  Run `rfbrowser init` in your shell.
    *   If `rfbrowser` isn't found, try `python -m Browser.entry init`.
    *   By default, it installs Chromium, Firefox, and WebKit. Use `--skip-browsers` to skip browser installation or specify browsers (e.g., `rfbrowser init firefox chromium`).

### Installing with Robotidy (optional)

For enhanced code formatting and maintainability:

*   `pip install robotframework-browser[tidy]`

## Update Instructions

1.  `pip install -U robotframework-browser`
2.  `rfbrowser clean-node`
3.  `rfbrowser init`

## Uninstall Instructions

1.  `rfbrowser clean-node`
2.  `pip uninstall robotframework-browser`

## Examples

### Robot Framework Example

```robotframework
*** Settings ***
Library   Browser

*** Test Cases ***
Example Test
    New Page    https://playwright.dev
    Get Text    h1    contains    Playwright
```

### Python Example

```python
import Browser
browser = Browser.Browser()
browser.new_page("https://playwright.dev")
assert 'Playwright' in browser.get_text("h1")
browser.close_browser()
```

### JavaScript Extension Example

```javascript
async function myGoToKeyword(url, page, logger) {
    logger("Going to " + url)
    return await page.goto(url);
}
myGoToKeyword.rfdoc = "This is my own go to keyword";
exports.__esModule = true;
exports.myGoToKeyword = myGoToKeyword;
```

```robotframework
*** Settings ***
Library   Browser  jsextension=${CURDIR}/mymodule.js

*** Test Cases ***
Example Test
   New Page
   myGoToKeyword   https://www.robotframework.org
```

### Further Examples

Explore more advanced usage, including selector syntax, browser page evaluation, asynchronous waiting, device descriptors, and HTTP requests, in the original [README](https://github.com/MarketSquare/robotframework-browser).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Core Team

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

This project is supported by Robocorp through the Robot Framework Foundation and driven by a vibrant community.  See the complete list of contributors in the original [README](https://github.com/MarketSquare/robotframework-browser).