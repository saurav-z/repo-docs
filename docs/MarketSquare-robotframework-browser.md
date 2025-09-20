# Robot Framework Browser Library: Modern Web Automation with Playwright

[Robot Framework Browser](https://github.com/MarketSquare/robotframework-browser) empowers you to automate web browsers with speed and reliability using the power of Playwright.

[![Version](https://img.shields.io/pypi/v/robotframework-browser.svg)](https://pypi.python.org/pypi/robotframework-browser)
[![Actions Status](https://github.com/MarketSquare/robotframework-browser/workflows/Continuous%20integration/badge.svg)](https://github.com/MarketSquare/robotframework-browser/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Key Features:

*   **Fast and Reliable Automation:** Built on Playwright for optimal speed and stability.
*   **Comprehensive Keyword Library:**  Offers a rich set of keywords for a wide range of web automation tasks.
*   **Ergonomic Selectors:**  Supports chaining of `text`, `css`, and `xpath` selectors for precise element targeting.
*   **JavaScript Integration:**  Easily extend functionality with custom JavaScript code within your tests.
*   **Asynchronous Operations:**  Handle HTTP requests and responses asynchronously for efficient testing.
*   **Device Emulation:**  Test across various devices with built-in device descriptors.
*   **HTTP Request Handling:** Send and parse HTTP requests directly within your test cases.
*   **Parallel Test Execution:** Integrate with Pabot for parallel test execution.
*   **Storage State Management:** Reuse authentication credentials, simplifying testing of authenticated areas.
*   **Robotidy Integration:** Transform deprecated keywords and apply automated formatting.

## Installation

### Prerequisites

*   Python 3.9 or newer
*   Node.js (LTS versions 20, 22, and 24 are supported) - Install from [https://nodejs.org/en/download/](https://nodejs.org/en/download/)

### Steps

1.  **Update pip:** `pip install -U pip`
2.  **Install the Library:** `pip install robotframework-browser`
3.  **Initialize Node Dependencies:** `rfbrowser init`  (or `python -m Browser.entry init`)
    *   By default, Chromium, Firefox, and WebKit browsers are installed.  Customize by using the `--skip-browsers`, `chromium`, `firefox` or `webkit` flags.  See original README for details.

    *   Alternatively, use the [docker images](https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable), see [docker/README.md](https://github.com/MarketSquare/robotframework-browser/blob/main/docker/README.md).

4.  **(Optional) Install with Robotidy support:** `pip install robotframework-browser[tidy]`

## Usage Examples

### Robot Framework

```robotframework
***Settings***
Library  Browser

***Test Cases***
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
***Settings***
Library   Browser  jsextension=${CURDIR}/mymodule.js

***Test Cases***
Example Test
   New Page
   myGoToKeyword   https://www.robotframework.org
```

## Update Instructions

1.  `pip install -U robotframework-browser`
2.  `rfbrowser clean-node`
3.  `rfbrowser init`

## Uninstall Instructions

1.  `rfbrowser clean-node`
2.  `pip uninstall robotframework-browser`

## Documentation and Resources

*   **Keyword Documentation:** [https://marketsquare.github.io/robotframework-browser/Browser.html](https://marketsquare.github.io/robotframework-browser/Browser.html)
*   **Project Website:** [https://robotframework-browser.org/](https://robotframework-browser.org/)
*   **Example Extension:** [https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015](https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015)
*   **Robotframework Browser Extensions:** [https://github.com/MarketSquare/robotframework-browser-extensions](https://github.com/MarketSquare/robotframework-browser-extensions)

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## Core Team

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

This project thrives on community contributions.
Supported by [Robocorp](https://robocorp.com/) through [Robot Framework Foundation](https://robotframework.org/foundation/).

[See the full list of contributors and their contributions.](https://github.com/MarketSquare/robotframework-browser/graphs/contributors)
```

Key improvements and SEO considerations:

*   **Clear Title and Hook:** The title is more descriptive and the first sentence acts as a strong hook, immediately conveying the library's purpose.
*   **Targeted Keywords:**  Keywords like "web automation," "Playwright," "Robot Framework," and specific features are strategically used throughout the text.
*   **Organized Headings:**  Uses clear headings and subheadings for better readability and SEO structure.
*   **Bulleted Feature List:** Highlights key features, making it easy for users to understand the library's capabilities.
*   **Concise and Actionable Instructions:**  Installation and usage instructions are clear and easy to follow.
*   **Internal and External Linking:** Includes links to the original repo, documentation, examples, and contributing guidelines to improve SEO and user experience.
*   **SEO-Friendly Formatting:** Uses Markdown for proper heading tags (H1, H2, etc.) which search engines use to index the page.
*   **Contributor Section:** Keeps the Contributor's list.
*   **Keywords in Meta Description:** This README will likely be the first thing people see and search engines will use this as the meta description.