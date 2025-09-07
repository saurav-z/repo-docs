# Robot Framework Browser Library

Automate your browser testing with ease using the [Robot Framework Browser library](https://github.com/MarketSquare/robotframework-browser), powered by Playwright for speed, reliability, and comprehensive visibility.

## Key Features

*   **Fast and Reliable:** Leverage the power of Playwright for efficient and dependable browser automation.
*   **Comprehensive Keyword Library:** Access a rich set of keywords for diverse automation needs, including page navigation, element interaction, and data validation.
*   **Ergonomic Selectors:** Utilize a flexible selector syntax with chaining for locating elements using text, CSS, and XPath.
*   **JavaScript Integration:** Seamlessly integrate JavaScript code within your tests for advanced functionality.
*   **Asynchronous Operations:** Easily handle asynchronous operations like waiting for HTTP requests and responses.
*   **Device Descriptor Support:** Test on different devices with pre-defined device descriptors.
*   **HTTP Request and Response Handling:** Send and parse HTTP requests within your tests.
*   **Parallel Execution:** Run tests in parallel using Pabot for faster execution.
*   **Reusable Authentication:** Utilize `Save Storage State` to reuse authentication credentials.
*   **Robotidy Integration:** Utilize the built in Robotidy transformers to update your code.

## Installation

1.  **Install Node.js:** Download and install Node.js from [https://nodejs.org/en/download/](https://nodejs.org/en/download/).
2.  **Update pip:** `pip install -U pip`
3.  **Install the library:** `pip install robotframework-browser`
4.  **Install node dependencies:** `rfbrowser init`
    *   If `rfbrowser` is not found, try `python -m Browser.entry init`

> Note: The installation process installs browser binaries (Chromium, Firefox, and WebKit) by default, adding around 700MB. You can skip the browser binaries installation with `rfbrowser init --skip-browsers`, but in this case, you'll be responsible for the browser binary installation. You can also install only select browsers by adding `chromium`, `firefox` or `webkit` as arguments to init command.

## Installation with Robotidy transformer

Install the library with Robotidy using:
`pip install robotframework-browser[tidy]`

Utilize the Robotidy transformers by running:
`rfbrowser transform --transformer-name /path/to/tests`
See full list of transformers: `rfbrowser transform --help`

## Update Instructions

1.  `pip install -U robotframework-browser`
2.  `rfbrowser clean-node`
3.  `rfbrowser init`

## Uninstall Instructions

1.  `rfbrowser clean-node`
2.  `pip uninstall robotframework-browser`

## Examples

### Robot Framework Test Case

```robotframework
*** Settings ***
Library   Browser

*** Test Cases ***
Example Test
    New Page    https://playwright.dev
    Get Text    h1    contains    Playwright
```

### Python Test Case

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

## Extending Browser Functionality

See [example](https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015).
Ready made extensions and a place to share your own at [robotframework-browser-extensions](https://github.com/MarketSquare/robotframework-browser-extensions).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.

## Documentation

*   [Keyword Documentation](https://marketsquare.github.io/robotframework-browser/Browser.html)
*   [Web Page](https://robotframework-browser.org/)

## Contributing

We welcome contributions!  This project is community driven.

## Core Team

*   Mikko Korpela
*   Tatu Aalto
*   Kerkko Pelttari
*   René Rohner

## Contributors

[See the full list of contributors](https://github.com/MarketSquare/robotframework-browser#contributors-)