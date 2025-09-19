# Robot Framework Browser Library: Modern Browser Automation

Automate web interactions with ease using the [Robot Framework Browser library](https://github.com/MarketSquare/robotframework-browser), powered by Playwright for speed, reliability, and comprehensive visibility.

## Key Features

*   **Fast and Reliable:** Leverage Playwright for rapid and dependable browser automation.
*   **Ergonomic Selectors:** Simplify element selection with intuitive text, CSS, and XPath selectors, including chaining capabilities.
*   **JavaScript Integration:** Extend functionality using custom JavaScript code within your tests.
*   **Asynchronous Operations:**  Handle HTTP requests and responses with asynchronous waiting capabilities.
*   **Device Descriptors:** Emulate various devices and screen sizes for responsive testing.
*   **HTTP Request/Response Handling:**  Send and parse HTTP requests and responses directly within your tests.
*   **Cross-Browser Support:**  Test across Chromium, Firefox, and WebKit.
*   **Robotidy Integration:**  Enhance code quality with optional Robotidy support for code formatting and transformation.
*   **Parallel Test Execution:**  Efficiently execute tests in parallel using Pabot.

## Installation

### Prerequisites

*   Python 3.9 or newer
*   Node.js (LTS versions 20, 22, and 24 are supported)

### Steps

1.  **Install Node.js:** Download and install from [https://nodejs.org/en/download/](https://nodejs.org/en/download/).
2.  **Update pip:** `pip install -U pip`
3.  **Install the library:** `pip install robotframework-browser`
4.  **Initialize Node dependencies:** `rfbrowser init`

    *   By default, this installs Chromium, Firefox, and WebKit. Customize the installation with `--skip-browsers` or by specifying browser names (e.g., `rfbrowser init firefox chromium`).

Or, use the [docker images](https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable)
documented at [docker/README.md](https://github.com/MarketSquare/robotframework-browser/blob/main/docker/README.md).

### Install with Robotidy (Optional)

Install with Robotidy by running `pip install robotframework-browser[tidy]` and use transformers like
`rfbrowser transform --wait-until-network-is-idle /path/to/tests` (see `rfbrowser transform --help` for details).

## Update

1.  `pip install -U robotframework-browser`
2.  `rfbrowser clean-node`
3.  `rfbrowser init`

## Uninstall

1.  `rfbrowser clean-node`
2.  `pip uninstall robotframework-browser`

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

## Parallel Test Execution (Pabot)

The library supports parallel execution using Pabot. Use the `ROBOT_FRAMEWORK_BROWSER_NODE_PORT` environment variable and the `spawn_node_process` helper (see the docs for the helper) to share node side RF Browser processes.

## Re-using Authentication Credentials

*   Use `Save Storage State` to save authentication information from local storage or cookies. See the documentation for usage examples.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Community

*   **Keyword Documentation:** [https://marketsquare.github.io/robotframework-browser/Browser.html](https://marketsquare.github.io/robotframework-browser/Browser.html)
*   **Web Page:** [https://robotframework-browser.org/](https://robotframework-browser.org/)
*   **Extensions:**  Share and discover extensions at [robotframework-browser-extensions](https://github.com/MarketSquare/robotframework-browser-extensions).

## Core Team

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

This project thrives on community contributions.  A huge thank you to all contributors!

<!-- ALL-CONTRIBUTORS-LIST:START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- Add Contributors section content from the original README here. -->
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->