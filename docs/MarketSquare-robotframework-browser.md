# Robot Framework Browser Library: Automate Your Web with Playwright

**Supercharge your Robot Framework tests with the power of Playwright, providing speed, reliability, and unmatched visibility into your web automation.** [Explore the original repository](https://github.com/MarketSquare/robotframework-browser).

[![Version](https://img.shields.io/pypi/v/robotframework-browser.svg)](https://pypi.python.org/pypi/robotframework-browser)
[![Actions Status](https://github.com/MarketSquare/robotframework-browser/workflows/Continuous%20integration/badge.svg)](https://github.com/MarketSquare/robotframework-browser/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Key Features:

*   **Fast and Reliable Automation:** Leverage Playwright for swift and dependable browser interaction.
*   **Comprehensive Keyword Library:** Access a rich set of keywords for diverse testing needs.
*   **Versatile Selector Syntax:** Utilize a user-friendly selector syntax, including chaining of `text`, `css`, and `xpath` selectors.
*   **JavaScript Extension:** Extend functionality with custom JavaScript code execution within the browser.
*   **Asynchronous Operations:** Support for asynchronous waiting for HTTP requests and responses.
*   **Device Emulation:** Easily test on different devices using device descriptors.
*   **HTTP Request Handling:** Send and parse HTTP requests and responses for comprehensive testing.
*   **Robotidy Integration:** Integrate with Robotidy for enhanced code formatting and transformation.

## Installation

**Prerequisites:**

*   Python 3.9 or newer
*   Node.js (LTS versions 20 and 22 are supported) - Install from [https://nodejs.org/en/download/](https://nodejs.org/en/download/)

**Installation Steps:**

1.  **Update pip:** `pip install -U pip`
2.  **Install the library:** `pip install robotframework-browser`
3.  **Initialize node dependencies:** `rfbrowser init`  (if `rfbrowser` is not found, try `python -m Browser.entry init`)

    *   By default, Chromium, Firefox, and WebKit browsers are installed. This can add a significant installation size (+700MB).
    *   Use `rfbrowser init --skip-browsers` to skip browser binary installation (you are responsible for their installation).
    *   Specify browser binaries to install only selected browsers: `rfbrowser init firefox chromium`

Or use the [docker images](https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable). Documented at [docker/README.md](https://github.com/MarketSquare/robotframework-browser/blob/main/docker/README.md).

### Install with transformer

Install library with Robotidy, run install with:
`pip install robotframework-browser[tidy]`. Starting from 18.3.0 release, library will provide external
Robotidy [transformer](https://robotidy.readthedocs.io/en/stable/external_transformers.html). Transformer provided
by Browser library can be run with command: `rfbrowser transform --transformer-name /path/to/tests`. Example:
`rfbrowser transform --wait-until-network-is-idle /path/to/tests` would transform deprecated `Wait Until Network Is Idle`
keyword to `Wait For Load State` keyword. To see full list of transformers provided by Browser library, run
command: `rfbrowser transform --help`.

## Update Instructions

1.  Update from commandline: `pip install -U robotframework-browser`
2.  Clean old node side dependencies and browser binaries: `rfbrowser clean-node`
3.  Install the node dependencies for the newly installed version: `rfbrowser init`

## Uninstall Instructions

1.  Clean old node side dependencies and browser binaries: `rfbrowser clean-node`
2.  Uninstall with pip: `pip uninstall robotframework-browser`

## Examples

### Robot Framework

```robotframework
*** Settings ***
Library   Browser

*** Test Cases ***
Example Test
    New Page    https://playwright.dev
    Get Text    h1    contains    Playwright
```

### Python

```python
import Browser
browser = Browser.Browser()
browser.new_page("https://playwright.dev")
assert 'Playwright' in browser.get_text("h1")
browser.close_browser()
```

### JavaScript Extension

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

[See more examples](https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015). Explore [robotframework-browser-extensions](https://github.com/MarketSquare/robotframework-browser-extensions) to share and use pre-built extensions.

### Parallel Test Execution

Leverage Pabot for parallel test execution.  Use the `ROBOT_FRAMEWORK_BROWSER_NODE_PORT` environment variable to share node processes, which can help with performance.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.

## Core Team

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

This project thrives thanks to its community!  ([See the full list of contributors](https://github.com/MarketSquare/robotframework-browser/graphs/contributors).)

Supported by [Robocorp](https://robocorp.com/) through [Robot Framework Foundation](https://robotframework.org/foundation/).