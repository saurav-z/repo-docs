<!-- Improved README.md -->
# Robot Framework Browser Library

[![Robot Framework Browser](https://github.com/MarketSquare/robotframework-browser/blob/main/docs/img/robotframework-browser-logo.png?raw=true)](https://github.com/MarketSquare/robotframework-browser)

**Automate your web testing with speed, reliability, and precision using the Robot Framework Browser library, powered by Playwright.**  This library provides a powerful and versatile solution for browser automation within the Robot Framework ecosystem. [Explore the original repository here](https://github.com/MarketSquare/robotframework-browser).

## Key Features

*   **Fast and Reliable Automation:** Leverage the speed and stability of Playwright for efficient testing.
*   **Cross-Browser Compatibility:**  Test across Chromium, Firefox, and WebKit.
*   **Ergonomic Selectors:**  Use intuitive syntax with chaining of text, CSS, and XPath selectors.
*   **JavaScript Integration:** Extend functionality with custom JavaScript code.
*   **Asynchronous Operations:** Seamlessly handle asynchronous HTTP requests and responses.
*   **Device Descriptors:** Simulate various devices for responsive testing.
*   **HTTP Request Handling:** Send and parse HTTP requests with ease.
*   **Parallel Test Execution:** Utilize Pabot for faster test execution.
*   **Extensive Documentation:** Comprehensive [keyword documentation](https://marketsquare.github.io/robotframework-browser/Browser.html) and a dedicated [web page](https://robotframework-browser.org/) are available.

## Installation

### Prerequisites

*   Python 3.9 or newer
*   Node.js (version 20, 22 or 24 LTS)

### Steps

1.  **Install Node.js:** Download and install from [https://nodejs.org/en/download/](https://nodejs.org/en/download/).
2.  **Update pip:** `pip install -U pip`
3.  **Install the library:** `pip install robotframework-browser`
4.  **Initialize node dependencies:** `rfbrowser init`
    *   If `rfbrowser` is not found, try `python -m Browser.entry init`.
    *   By default, Chromium, Firefox, and WebKit browsers are installed, which adds roughly 700MB to the installation size.
    *   You can skip browser installation by running  `rfbrowser init --skip-browsers`, but you will then need to install the browsers yourself.
    *   Specific browsers can be installed using `rfbrowser init firefox` or `rfbrowser init firefox chromium`.
5.  **Docker:** Utilize pre-built [docker images](https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable).  See [docker/README.md](https://github.com/MarketSquare/robotframework-browser/blob/main/docker/README.md) for documentation.

### Installation with Robotidy

To install with Robotidy, include the `[tidy]` extra: `pip install robotframework-browser[tidy]`

## Updating the Library

1.  **Update the library:** `pip install -U robotframework-browser`
2.  **Clean old node dependencies:** `rfbrowser clean-node`
3.  **Initialize node dependencies:** `rfbrowser init`

## Uninstalling the Library

1.  **Clean old node dependencies:** `rfbrowser clean-node`
2.  **Uninstall the library:** `pip uninstall robotframework-browser`

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

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.

## Core Team

(In order of appearance)

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

This project thrives on community contributions, with support from [Robocorp](https://robocorp.com/) through the [Robot Framework Foundation](https://robotframework.org/foundation/).