# Robot Framework Browser Library: Modern Browser Automation

**Automate your web testing with speed, reliability, and enhanced visibility using the Robot Framework Browser library powered by Playwright!**

[Explore the original repository](https://github.com/MarketSquare/robotframework-browser)

## Key Features

*   🚀 **Blazing Fast Automation:** Leverage Playwright's speed for efficient test execution.
*   ✅ **Reliable Testing:** Minimize flakiness and ensure consistent results.
*   🔍 **Enhanced Visibility:** Gain deeper insights into your tests with detailed reporting and debugging tools.
*   🔌 **Flexible Selector Syntax:** Utilize ergonomic selector syntax, including chaining for text, CSS, and XPath selectors.
*   💻 **JavaScript Integration:** Extend functionality with custom JavaScript, and share your own extensions.
*   📱 **Device Descriptors:** Simulate various devices for responsive design testing.
*   📡 **HTTP Request Handling:** Send and parse HTTP requests within your tests.
*   🔄 **Asynchronous Operations:** Easily handle asynchronous events like network requests.
*   💾 **Storage State Management:** Save and reuse authentication credentials, local storage, and cookies.
*   📦 **Robotidy Integration:** Use [Robotidy](https://robotidy.readthedocs.io/en/stable/) to automatically transform and enhance your Robot Framework tests.
*   🐳 **Docker Support:** Easy setup and environment consistency through [docker images](https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable).

## Installation

### Prerequisites

*   Python 3.9 or newer
*   Node.js (20, 22, and 24 LTS versions are supported) -  Install node.js from [https://nodejs.org/en/download/](https://nodejs.org/en/download/)

### Steps

1.  **Update pip:** `pip install -U pip`
2.  **Install the Library:** `pip install robotframework-browser`
3.  **Initialize Node Dependencies:** `rfbrowser init`
    *   If `rfbrowser` is not found, try `python -m Browser.entry init`
    *   By default, Chromium, Firefox, and WebKit browsers are installed.  Use `rfbrowser init --skip-browsers` to skip browser installation, or specify browser binaries (e.g., `rfbrowser init firefox chromium`).

### Using Docker

Refer to the [docker/README.md](https://github.com/MarketSquare/robotframework-browser/blob/main/docker/README.md) for docker images.

## Installation with Robotidy

If you want to use Robotidy as well:

1.  `pip install robotframework-browser[tidy]`
2.  Transform the deprecated `Wait Until Network Is Idle` keyword to `Wait For Load State` keyword with command: `rfbrowser transform --wait-until-network-is-idle /path/to/tests`

## Updating

1.  `pip install -U robotframework-browser`
2.  `rfbrowser clean-node`
3.  `rfbrowser init`

## Uninstalling

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

## Further Resources

*   [Keyword Documentation](https://marketsquare.github.io/robotframework-browser/Browser.html)
*   [Homepage](https://robotframework-browser.org/)
*   [Example Extensions](https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015)
*   [Share Extensions](https://github.com/MarketSquare/robotframework-browser-extensions)

## Development

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.

## Core Team

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

[List of Contributors](https://github.com/MarketSquare/robotframework-browser#contributors-)