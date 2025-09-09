# Robot Framework Browser Library

Automate your web browser testing with the power of Playwright and Robot Framework.  This library provides a robust and reliable solution for modern web automation, offering speed, reliability, and clear visibility into your testing processes.  Visit the [Robot Framework Browser Library GitHub repository](https://github.com/MarketSquare/robotframework-browser) for more information.

## Key Features

*   **Cross-Browser Support:** Test across Chromium, Firefox, and WebKit.
*   **Ergonomic Selectors:**  Simplify element selection with intuitive syntax, supporting chaining of text, CSS, and XPath selectors.
*   **JavaScript Integration:** Extend functionality with custom JavaScript code execution within the browser.
*   **Asynchronous Operations:**  Easily handle asynchronous tasks like waiting for HTTP requests and responses.
*   **Device Emulation:** Test on different devices with device descriptor support.
*   **HTTP Request Handling:**  Send and parse HTTP requests directly from your tests.
*   **Parallel Test Execution:**  Run tests concurrently using Pabot for faster execution.
*   **Authentication Support:** Re-use authentication credentials using `Save Storage State` functionality.
*   **Robotidy Integration:** Includes an optional dependency with Robotidy, to transform deprecated keywords.

## Installation

1.  **Prerequisites:** Ensure you have Python 3.9 or newer and Node.js (versions 20 and 22 LTS recommended) installed.
2.  **Update pip:** `pip install -U pip`
3.  **Install Library:** `pip install robotframework-browser`
4.  **Initialize Node Dependencies:** Run `rfbrowser init`

    *   If `rfbrowser` is not found, try `python -m Browser.entry init`
    *   You can skip browser binaries installation by using `rfbrowser init --skip-browsers`.
    *   You can install specific browsers with `rfbrowser init firefox chromium`.

## Installation with Robotidy

To install with Robotidy, use `pip install robotframework-browser[tidy]`. Use the transformer with the command: `rfbrowser transform --transformer-name /path/to/tests`.  See all transformers by running the command: `rfbrowser transform --help`.

## Update Instructions

1.  `pip install -U robotframework-browser`
2.  `rfbrowser clean-node`
3.  `rfbrowser init`

## Uninstall Instructions

1.  `rfbrowser clean-node`
2.  `pip uninstall robotframework-browser`

## Examples

**Robot Framework Example:**

```robotframework
*** Settings ***
Library   Browser

*** Test Cases ***
Example Test
    New Page    https://playwright.dev
    Get Text    h1    contains    Playwright
```

**Python Example:**

```python
import Browser
browser = Browser.Browser()
browser.new_page("https://playwright.dev")
assert 'Playwright' in browser.get_text("h1")
browser.close_browser()
```

## Development

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on contributing to the project.

## Core Team
*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

This project is community driven and supported by [Robocorp](https://robocorp.com/) through [Robot Framework Foundation](https://robotframework.org/foundation/). (See the full list of contributors in the original README)