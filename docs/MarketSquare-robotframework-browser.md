# Robot Framework Browser Library

[Robot Framework Browser](https://github.com/MarketSquare/robotframework-browser) simplifies web browser automation with the power of Playwright, offering speed, reliability, and unparalleled visibility for your testing needs.

## Key Features:

*   **High Performance:** Achieve blazing-fast browser automation with Playwright.
*   **Reliable Execution:** Ensure stable and dependable tests.
*   **Enhanced Visibility:** Gain detailed insights into your test runs.
*   **Ergonomic Selectors:** Use a natural and intuitive selector syntax, supporting chaining of `text`, `css`, and `xpath` selectors.
*   **Asynchronous Operations:** Easily handle asynchronous tasks like waiting for HTTP requests and responses.
*   **Device Emulation:** Test across various devices with built-in device descriptors.
*   **HTTP Request/Response Handling:**  Send and parse HTTP requests within your tests.
*   **JavaScript Integration:** Extend your tests with custom JavaScript code.
*   **Robotidy Integration**: Integrates with Robotidy for automatic code formatting and transformation.

## Installation

### Prerequisites

*   Node.js (version 20 or 22 LTS recommended) - [Download from here](https://nodejs.org/en/download/)
*   Python 3.9 or newer
*   Ensure you have the latest version of pip: `pip install -U pip`

### Steps

1.  Install the library using pip: `pip install robotframework-browser`
2.  Initialize node dependencies: `rfbrowser init` (If `rfbrowser` is not found, try `python -m Browser.entry init`)

    *   By default, this command installs Chromium, Firefox, and WebKit browsers. You can skip browser installation with `rfbrowser init --skip-browsers` or specify particular browsers, e.g. `rfbrowser init firefox chromium`.

### Installation with Robotidy

To install with Robotidy: `pip install robotframework-browser[tidy]`

## Update Instructions

1.  Update the library: `pip install -U robotframework-browser`
2.  Clean old node dependencies: `rfbrowser clean-node`
3.  Reinstall dependencies: `rfbrowser init`

## Uninstall Instructions

1.  Clean old node dependencies and browser binaries: `rfbrowser clean-node`
2.  Uninstall the library: `pip uninstall robotframework-browser`

## Examples

### Basic Test Case (Robot Framework)

```robotframework
*** Settings ***
Library   Browser

*** Test Cases ***
Example Test
    New Page    https://playwright.dev
    Get Text    h1    contains    Playwright
```

### Basic Test Case (Python)

```python
import Browser
browser = Browser.Browser()
browser.new_page("https://playwright.dev")
assert 'Playwright' in browser.get_text("h1")
browser.close_browser()
```

### JavaScript Extension

```robotframework
*** Settings ***
Library   Browser  jsextension=${CURDIR}/mymodule.js

*** Test Cases ***
Example Test
   New Page
   myGoToKeyword   https://www.robotframework.org
```

(See [example](https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015). Ready made extensions and a place to share your own at [robotframework-browser-extensions](https://github.com/MarketSquare/robotframework-browser-extensions).)

## Advanced Features:

*   **Ergonomic Selector Syntax:** Easily select elements using chained selectors like  `Click    "Login" >> xpath=../input`
*   **Browser Page Evaluation:** Use `Evaluate JavaScript` to interact with page elements in-browser.
*   **Asynchronous Waiting:**  Leverage `Promise To` and `Wait For Response` for handling asynchronous operations.
*   **Device Descriptors:** Test on various devices using `New Context  &{device}` with defined device dictionaries.
*   **HTTP Request Handling:** Make HTTP requests and parse responses with the `HTTP` keyword.
*   **Parallel test execution:** Utilize pabot for parallel test execution.
*   **Re-using authentication credentials:** `Save Storage State` and `Load Storage State` can be utilized to share session cookies between different test cases and suites.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.

## Contributors

This project thrives thanks to its community of contributors.  A big thank you to all contributors! (See the full list in the original README).

---

**[Get started with Robot Framework Browser today!](https://github.com/MarketSquare/robotframework-browser)**
```
Key improvements and SEO considerations:

*   **Clear, concise introduction:** Sets the stage and highlights key benefits.
*   **Keyword-rich headings:** Uses relevant terms (e.g., "Robot Framework Browser," "Playwright," "Browser Automation") for SEO.
*   **Bulleted feature list:**  Easy for users to scan and understand key capabilities.
*   **Action-oriented language:** Encourages users to take action (e.g., "Get started," "Install").
*   **Links to key resources:** Directs users to the repository and other relevant documentation.
*   **Concise Installation, Update, and Uninstall Sections:** Provides clear and to-the-point instructions.
*   **Combined and condensed example section** Offers concise examples in both Robot Framework and Python.
*   **Contributor section retained** Acknowledges the valuable community contributors.