# Robot Framework Browser Library: Automate Web Testing with Ease

[Robot Framework Browser](https://github.com/MarketSquare/robotframework-browser) empowers testers to create robust and reliable web automation tests using the popular Robot Framework and the powerful Playwright library.

## Key Features:

*   **Fast and Reliable Automation:** Achieve high-speed and dependable testing with Playwright.
*   **Intuitive Syntax:** Utilize a user-friendly keyword-driven approach for easy test creation and maintenance.
*   **Cross-Browser Compatibility:** Test across Chromium, Firefox, and WebKit, ensuring broad coverage.
*   **Ergonomic Selectors:** Leverage advanced selector syntax, including chaining `text`, `css`, and `xpath` for precise element targeting.
*   **JavaScript Integration:**  Extend functionality with custom JavaScript snippets and extensions.
*   **Asynchronous Operations:** Seamlessly handle asynchronous HTTP requests and responses.
*   **Device Descriptors:** Test with different device emulations.
*   **HTTP Request & Response Handling:** Send HTTP requests and parse responses directly within your tests.
*   **Parallel Test Execution:** Execute tests concurrently using Pabot for faster results.
*   **Robotidy Integration:**  Utilize Robotidy transformers for code standardization and automation.

## Installation

Follow these steps to install and configure the Robot Framework Browser library:

1.  **Prerequisites:** Ensure you have Python 3.9 or newer and Node.js (LTS versions 20, 22, and 24 are supported) installed.
2.  **Update pip:** `pip install -U pip`
3.  **Install Library:** `pip install robotframework-browser`
4.  **Initialize Node Dependencies:** Run `rfbrowser init` (or `python -m Browser.entry init` if `rfbrowser` is not found).  This installs browser binaries.
    *   Use `rfbrowser init --skip-browsers` to skip browser binary installation (you'll need to manage the binaries yourself).
    *   Specify browsers with: `rfbrowser init firefox chromium`

## Usage Examples

### Basic Test Case

```robotframework
*** Settings ***
Library   Browser

*** Test Cases ***
Example Test
    New Page    https://playwright.dev
    Get Text    h1    contains    Playwright
```

### Python Integration

```python
import Browser
browser = Browser.Browser()
browser.new_page("https://playwright.dev")
assert 'Playwright' in browser.get_text("h1")
browser.close_browser()
```

### JavaScript Extension

```javascript
// mymodule.js
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

##  [Visit the Official Documentation](https://marketsquare.github.io/robotframework-browser/) for comprehensive keyword documentation and [robotframework-browser.org](https://robotframework-browser.org/) for more details.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.

## Core Team

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Community

Special thanks to our valuable contributors, supported by [Robocorp](https://robocorp.com/) through [Robot Framework Foundation](https://robotframework.org/foundation/).
*(See the original README for the extensive list of contributors)*