# Robot Framework Browser Library: Modern Browser Automation

[Robot Framework Browser](https://github.com/MarketSquare/robotframework-browser) is a powerful library that brings the speed and reliability of [Playwright](https://playwright.dev/) to Robot Framework, revolutionizing browser automation.

## Key Features:

*   **Fast and Reliable:** Leveraging Playwright for efficient and dependable browser testing.
*   **Cross-Browser Support:** Works seamlessly with Chromium, Firefox, and WebKit.
*   **Ergonomic Selectors:** Simplifies element selection with chained `text`, `css`, and `xpath` selectors.
*   **JavaScript Integration:** Enables extension of functionality with custom JavaScript code.
*   **Asynchronous Operations:** Provides capabilities for asynchronously waiting for HTTP requests and responses.
*   **Device Emulation:** Offers device descriptors for emulating different screen sizes and devices.
*   **HTTP Request Handling:** Allows sending and parsing HTTP requests and responses.
*   **Extensible with Robotidy:**  Integrates with Robotidy for code formatting and transformation.

## Installation

Follow these steps to get started:

1.  **Install Node.js:** Download and install Node.js from [https://nodejs.org/en/download/](https://nodejs.org/en/download/).
2.  **Update pip:** `pip install -U pip`
3.  **Install the Library:** `pip install robotframework-browser`
4.  **Initialize Node Dependencies:** Run `rfbrowser init` in your shell to install necessary node dependencies and browser binaries.
    *   You can skip browser binary installation with `rfbrowser init --skip-browsers`.
    *   Use `rfbrowser init <browser>` (e.g., `rfbrowser init firefox`) to install specific browser binaries.

## Upgrade and Uninstall

*   **Upgrade:**
    1.  `pip install -U robotframework-browser`
    2.  `rfbrowser clean-node`
    3.  `rfbrowser init`
*   **Uninstall:**
    1.  `rfbrowser clean-node`
    2.  `pip uninstall robotframework-browser`

## Examples

### Simple Test Case
```robotframework
*** Settings ***
Library   Browser

*** Test Cases ***
Example Test
    New Page    https://playwright.dev
    Get Text    h1    contains    Playwright
```

### Python Usage
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

## Advanced Features:

*   **Ergonomic Selector Syntax:** Supports chaining of `text`, `css`, and `xpath` selectors.
*   **Evaluate in Browser Page:** Execute JavaScript code directly within the browser context.
*   **Asynchronous Waiting:**  Asynchronously wait for HTTP requests and responses.
*   **Device Descriptors:** Emulate different devices with pre-defined configurations.
*   **HTTP Requests:** Send HTTP requests and parse responses directly.
*   **Authentication:**  Save and reuse authentication credentials using `Save Storage State`.
*   **Parallel Test Execution:** Leverage Pabot for parallel test execution.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.

## Core Team

In order of appearance.

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

A huge thank you to all our contributors! See the full list [here](https://github.com/MarketSquare/robotframework-browser#contributors-).

---
```

**Key improvements and SEO Optimizations:**

*   **Concise Hook:** The opening sentence immediately grabs attention and highlights the core benefit.
*   **Keyword Optimization:**  Included relevant keywords like "Robot Framework," "browser automation," "Playwright," "browser testing," and key features throughout the headings and text.
*   **Clear Headings:** Uses clear, descriptive headings for easy navigation and improved SEO.
*   **Bulleted Key Features:** Makes it easy for users to scan and understand the library's capabilities.
*   **Simplified Installation:** Installation instructions are streamlined and user-friendly.
*   **Concise Examples:** Examples are well-formatted, immediately demonstrating the library's usage.
*   **Link Back to Original Repo:**  The initial link is maintained.
*   **Call to Action:** Encourages exploration of advanced features.
*   **Enhanced Readability:** The use of bullet points and subheadings makes it easy to read and digest.
*   **Concise descriptions**: shortened explanations for a quicker scan of the content.