# Robot Framework Browser Library

[![PyPI Version](https://img.shields.io/pypi/v/robotframework-browser.svg)](https://pypi.python.org/pypi/robotframework-browser)
[![Actions Status](https://github.com/MarketSquare/robotframework-browser/workflows/Continuous%20integration/badge.svg)](https://github.com/MarketSquare/robotframework-browser/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Automate your web testing with speed, reliability, and visibility using the Robot Framework Browser library, powered by Playwright.**  [Get started on GitHub!](https://github.com/MarketSquare/robotframework-browser)

## Key Features

*   **Robust Browser Automation:** Leverage the power of Playwright for reliable and efficient web automation.
*   **Ergonomic Selectors:**  Simplify element selection with a user-friendly selector syntax, including chaining of `text`, `css`, and `xpath` selectors.
*   **JavaScript Integration:** Easily extend functionality by integrating JavaScript code within your tests.
*   **Asynchronous Operations:**  Handle asynchronous requests and responses with ease, improving test accuracy and speed.
*   **Device Descriptor Support:** Test across various devices using pre-defined device descriptors.
*   **HTTP Request Capabilities:**  Send and parse HTTP requests directly within your tests.
*   **Parallel Test Execution:** Utilize Pabot for parallel test execution, significantly reducing test execution time.
*   **Reusable Authentication:**  Save and reuse authentication credentials, such as cookies and local storage.
*   **Rich Documentation:** Comprehensive [keyword documentation](https://marketsquare.github.io/robotframework-browser/Browser.html) and a detailed [web page](https://robotframework-browser.org/) are available.

## Installation

1.  **Prerequisites:** Ensure you have Python 3.9 or newer and Node.js installed.
2.  **Update pip:** `pip install -U pip`
3.  **Install the library:** `pip install robotframework-browser`
4.  **Initialize node dependencies:** `rfbrowser init` (or `python -m Browser.entry init` if `rfbrowser` is not found)

    *   You can skip browser binary installation with `rfbrowser init --skip-browsers`, but you will then need to install the browsers.
    *   Install specific browsers: `rfbrowser init firefox chromium`

    *   Use [docker images](https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable) documented at [docker/README.md](https://github.com/MarketSquare/robotframework-browser/blob/main/docker/README.md).

## Installation with Robotidy transformer

Starting from release 18.3.0 Browser library has optional dependency with
[Robotidy](https://robotidy.readthedocs.io/en/stable/). Install library with Robotidy, run install with:
`pip install robotframework-browser[tidy]`. Starting from 18.3.0 release, library will provide external
Robotidy [transformer](https://robotidy.readthedocs.io/en/stable/external_transformers.html). Transformer provided
by Browser library can be run with command: `rfbrowser transform --transformer-name /path/to/tests`. Example:
`rfbrowser transform --wait-until-network-is-idle /path/to/tests` would transform deprecated `Wait Until Network Is Idle`
keyword to `Wait For Load State` keyword. To see full list of transformers provided by Browser library, run
command: `rfbrowser transform --help`.

## Updating

1.  `pip install -U robotframework-browser`
2.  `rfbrowser clean-node`
3.  `rfbrowser init`

## Uninstalling

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

For more examples, see the [example](https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015) and  [robotframework-browser-extensions](https://github.com/MarketSquare/robotframework-browser-extensions).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.

## Core Team

(Listed in order of appearance in the original README)

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

This project is community-driven, and we appreciate all contributions. Thank you to all the contributors!  (See the full list in the original README.)

*Supported by [Robocorp](https://robocorp.com/) through [Robot Framework Foundation](https://robotframework.org/foundation/).*