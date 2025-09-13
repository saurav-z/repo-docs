# Robot Framework Browser Library: Automate Your Web Tests with Ease

[Robot Framework Browser](https://github.com/MarketSquare/robotframework-browser) is a powerful, open-source library that simplifies web automation testing using the Robot Framework. Leveraging the capabilities of Playwright, this library empowers you to create fast, reliable, and highly visible browser automation tests.

## Key Features

*   **Fast & Reliable Automation:** Built on Playwright for blazing-fast test execution and robust performance.
*   **Comprehensive Keyword Library:** Offers a rich set of keywords for interacting with web pages, including element selection, navigation, and data validation.
*   **JavaScript Integration:**  Easily extend your tests with custom JavaScript code for advanced scenarios.
*   **Ergonomic Selectors:** Simplify element targeting with an intuitive selector syntax, supporting chaining of text, CSS, and XPath selectors.
*   **Asynchronous Operations:** Supports asynchronous waiting for HTTP requests and responses, enabling testing of complex web applications.
*   **Device Emulation:**  Test your web applications on different devices using built-in device descriptors.
*   **HTTP Request Handling:** Send and parse HTTP requests directly from your tests, perfect for API testing and data validation.
*   **Docker Support:** Easily set up and run your tests within Docker containers.
*   **Robotidy Integration:** (Optional) Enhance code quality and maintainability with the Robotidy transformer.

## Installation

Follow these steps to get started:

1.  **Prerequisites:** Ensure you have Python 3.9 or newer and Node.js (version 20 or 22 LTS) installed.
2.  **Update Pip:** `pip install -U pip` (to ensure the latest version).
3.  **Install the Library:** `pip install robotframework-browser`
4.  **Initialize Node Dependencies:** Run `rfbrowser init` to install the necessary node dependencies. If `rfbrowser` is not found, try `python -m Browser.entry init`.

    *   By default, Chromium, Firefox, and WebKit browsers are installed. You can skip this with `rfbrowser init --skip-browsers` or specify particular browsers e.g. `rfbrowser init firefox`.
    *   Alternatively, leverage the pre-built [Docker images](https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable) - see [docker/README.md](https://github.com/MarketSquare/robotframework-browser/blob/main/docker/README.md) for instructions.
5.  **(Optional) Install Robotidy:** For integration with Robotidy, use `pip install robotframework-browser[tidy]`.

## Examples

### Basic Test Case

```robotframework
*** Settings ***
Library   Browser

*** Test Cases ***
Example Test
    New Page    https://playwright.dev
    Get Text    h1    contains    Playwright
```

### Extending with Python

```python
import Browser
browser = Browser.Browser()
browser.new_page("https://playwright.dev")
assert 'Playwright' in browser.get_text("h1")
browser.close_browser()
```

### Extending with JavaScript

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

## Additional Resources

*   **Keyword Documentation:** [Detailed keyword documentation](https://marketsquare.github.io/robotframework-browser/Browser.html).
*   **Web Page:** [Official project website](https://robotframework-browser.org/).
*   **Example Extensions:** [robotframework-browser-extensions](https://github.com/MarketSquare/robotframework-browser-extensions).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.

## Contributors

This project thrives thanks to the contributions of a vibrant community.
Special thanks to [Robocorp](https://robocorp.com/) through [Robot Framework Foundation](https://robotframework.org/foundation/) for their support.

[List of Contributors with badges - original README]

---