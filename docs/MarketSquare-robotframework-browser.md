# Robot Framework Browser Library: Automate Your Web Testing with Ease

[Robot Framework Browser](https://github.com/MarketSquare/robotframework-browser) is a powerful library built on Playwright, revolutionizing browser automation for Robot Framework. It provides speed, reliability, and visibility to take your web testing to the next level.

## Key Features

*   **Fast and Reliable:** Leverage the power of Playwright for efficient and dependable browser automation.
*   **Comprehensive Keyword Library:** Access a rich set of keywords to control browsers, interact with web elements, and manage network traffic.
*   **Cross-Browser Support:** Test on Chromium, Firefox, and WebKit with consistent results.
*   **Ergonomic Selectors:** Simplify element selection with intuitive text, CSS, and XPath selectors, including selector chaining.
*   **JavaScript Integration:** Extend your tests with custom JavaScript code execution within the browser.
*   **Asynchronous Operations:** Easily handle asynchronous requests and responses with built-in support for promises and waiting.
*   **Device Descriptor Support:** Emulate various devices, screen sizes and orientations for responsive testing.
*   **HTTP Request/Response Handling:** Send and receive HTTP requests and parse their responses.
*   **Parallel Test Execution:** Integrate with Pabot to execute tests in parallel and speed up test runs.
*   **Authentication Handling:**  Leverage `Save Storage State` keyword for handling authentication across your tests.

## Installation

1.  **Prerequisites:** Ensure you have Python 3.9 or newer and Node.js (versions 20 and 22 LTS) installed.
2.  **Update pip:**  `pip install -U pip`
3.  **Install the Library:** `pip install robotframework-browser`
4.  **Initialize Node Dependencies:** `rfbrowser init` (or `python -m Browser.entry init` if `rfbrowser` is not found)

    *   By default, Chromium, Firefox, and WebKit browsers are installed. Skip with `rfbrowser init --skip-browsers` if you have them installed separately. You can specify the browser binaries to install:  `rfbrowser init firefox chromium`.
    *   Alternatively, use the [Docker images](https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable).

## Advanced Features

### Install with transformer

Starting from release 18.3.0 Browser library has optional dependency with
[Robotidy](https://robotidy.readthedocs.io/en/stable/). Install library with Robotidy, run install with:
`pip install robotframework-browser[tidy]`. Starting from 18.3.0 release, library will provide external
Robotidy [transformer](https://robotidy.readthedocs.io/en/stable/external_transformers.html). Transformer provided
by Browser library can be run with command: `rfbrowser transform --transformer-name /path/to/tests`. Example:
`rfbrowser transform --wait-until-network-is-idle /path/to/tests` would transform deprecated `Wait Until Network Is Idle`
keyword to `Wait For Load State` keyword. To see full list of transformers provided by Browser library, run
command: `rfbrowser transform --help`.

### Update Instructions

1.  `pip install -U robotframework-browser`
2.  `rfbrowser clean-node`
3.  `rfbrowser init`

### Uninstall Instructions

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

For more examples, see the [examples directory](https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015).

## Development

Consult [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Contributors

This project thrives thanks to the contributions of many.  See the [full list of contributors](#contributors-)
and the [Robocorp](https://robocorp.com/) support via the [Robot Framework Foundation](https://robotframework.org/foundation/).

```