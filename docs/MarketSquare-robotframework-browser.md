# Robot Framework Browser Library: Automate Web Testing with Ease

Tired of tedious browser automation?  The **Robot Framework Browser Library**, built on Playwright, brings speed, reliability, and visibility to your web testing workflows. Check it out on [GitHub](https://github.com/MarketSquare/robotframework-browser).

## Key Features

*   **Powered by Playwright:** Leverage the power of Playwright for robust and modern browser automation.
*   **Fast and Reliable:** Achieve blazing-fast test execution and dependable results.
*   **Comprehensive Keyword Documentation:**  Explore a rich set of keywords for various automation tasks.  Access the [keyword documentation](https://marketsquare.github.io/robotframework-browser/Browser.html)
*   **Ergonomic Selectors:**  Use an intuitive selector syntax with chaining for efficient element identification.
*   **JavaScript Extension:** Extend functionality with custom JavaScript code, seamlessly integrated into your tests.
*   **Asynchronous Operations:** Handle asynchronous HTTP requests and responses with ease.
*   **Device Descriptors:**  Test across various devices and screen sizes using device descriptors.
*   **HTTP Request Support:**  Send and parse HTTP requests directly within your tests.
*   **Parallel Testing:**  Execute tests concurrently using Pabot for faster test runs.
*   **Extendable & Customizable:**  Easily create custom keywords and share your extensions for maximum flexibility.
*   **Authentication Handling:**  Use `Save Storage State` for handling authentication in your test suites.

## Installation

### Prerequisites:

*   Python 3.9 or newer
*   Node.js (version 20 and 22 LTS versions supported)

### Steps:

1.  **Install Node.js:** Download and install from [https://nodejs.org/en/download/](https://nodejs.org/en/download/).
2.  **Update pip:** `pip install -U pip`
3.  **Install the library:** `pip install robotframework-browser`
4.  **Initialize Node Dependencies:**  Run `rfbrowser init`
    *   If `rfbrowser` is not found, try `python -m Browser.entry init`
    *   By default, Chromium, Firefox, and WebKit browsers are installed.
    *   Use `--skip-browsers` with `rfbrowser init --skip-browsers` to skip browser binary installation. You will then be responsible for browser binary installation.
    *   You can specify specific browsers: `rfbrowser init firefox` or `rfbrowser init firefox chromium`

### Using Docker (Alternative Installation)

*   Use the official docker images: See [docker/README.md](https://github.com/MarketSquare/robotframework-browser/blob/main/docker/README.md) and the [package registry](https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable).

## Installation with Robotidy transformer

*   Optional Robotidy dependency. Install with `pip install robotframework-browser[tidy]`
*   Use Browser library's external transformer: `rfbrowser transform --transformer-name /path/to/tests`
*   To see transformer options: `rfbrowser transform --help`

## Updating

1.  `pip install -U robotframework-browser`
2.  `rfbrowser clean-node`
3.  `rfbrowser init`

## Uninstalling

1.  `rfbrowser clean-node`
2.  `pip uninstall robotframework-browser`

## Examples

### Robot Framework Example
```RobotFramework
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
```JavaScript
async function myGoToKeyword(url, page, logger) {
    logger("Going to " + url)
    return await page.goto(url);
}
myGoToKeyword.rfdoc = "This is my own go to keyword";
exports.__esModule = true;
exports.myGoToKeyword = myGoToKeyword;
```

```RobotFramework
*** Settings ***
Library   Browser  jsextension=${CURDIR}/mymodule.js

*** Test Cases ***
Example Test
   New Page
   myGoToKeyword   https://www.robotframework.org
```

Find more examples in the [example directory](https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015) and share your extensions at [robotframework-browser-extensions](https://github.com/MarketSquare/robotframework-browser-extensions).

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

This project thrives thanks to its contributors, supported by [Robocorp](https://robocorp.com/) through [Robot Framework Foundation](https://robotframework.org/foundation/).
(See the full list in the original README).