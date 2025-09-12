# Robot Framework Browser Library

**Automate your web testing with speed, reliability, and visibility using the Robot Framework Browser library, powered by Playwright!**  Explore the power of end-to-end testing with this robust and feature-rich library.  [Check out the original repo](https://github.com/MarketSquare/robotframework-browser).

## Key Features

*   **Effortless Installation:** Easily install the library with a simple pip command.
*   **Cross-Browser Support:**  Automate tests across Chromium, Firefox, and WebKit.
*   **Ergonomic Selectors:**  Use a powerful selector syntax that supports chaining of `text`, `css`, and `xpath` selectors.
*   **JavaScript Integration:** Seamlessly extend your tests with JavaScript for advanced functionality.
*   **Asynchronous Operations:**  Efficiently handle asynchronous waiting for HTTP requests and responses.
*   **Device Emulation:**  Test on different devices with pre-configured device descriptors.
*   **HTTP Request Handling:** Send and parse HTTP requests and responses within your tests.
*   **Parallel Execution:**  Leverage Pabot for parallel test execution and faster results.
*   **Re-using authentication credentials:** Save Storage State to reuse authentication credentials.
*   **Robotidy Integration:** Robotidy Transformer to transform deprecated keywords.

## Installation Guide

### Prerequisites

*   Python 3.9 or newer
*   Node.js (version 20 or 22 LTS recommended) - install from [https://nodejs.org/en/download/](https://nodejs.org/en/download/)

### Steps

1.  **Update pip:** `pip install -U pip`
2.  **Install the library:** `pip install robotframework-browser`
3.  **Initialize Node dependencies:** `rfbrowser init`
    *   If `rfbrowser` is not found, try `python -m Browser.entry init`
    *   Use `--skip-browsers` to skip browser binary installation. You'll then need to install browser binaries manually.
    *   You can install specific browsers: `rfbrowser init firefox`, `rfbrowser init chromium firefox`.

### Using Docker

Utilize pre-built Docker images for easy setup.  See the [docker/README.md](https://github.com/MarketSquare/robotframework-browser/blob/main/docker/README.md) for details.

## Advanced Features

*   **Transformer with Robotidy:** To install library with Robotidy, run `pip install robotframework-browser[tidy]`. Use `rfbrowser transform --transformer-name /path/to/tests` to transform deprecated keywords. Get help with `rfbrowser transform --help`
*   **Example Usage:**

    *   **Robot Framework:**

    ```robotframework
    *** Settings ***
    Library   Browser

    *** Test Cases ***
    Example Test
        New Page    https://playwright.dev
        Get Text    h1    contains    Playwright
    ```

    *   **Python:**

    ```python
    import Browser
    browser = Browser.Browser()
    browser.new_page("https://playwright.dev")
    assert 'Playwright' in browser.get_text("h1")
    browser.close_browser()
    ```

    *   **JavaScript Extension:**

    ```JavaScript
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

    See more examples at [robotframework-browser-extensions](https://github.com/MarketSquare/robotframework-browser-extensions).

## Updating the Library

1.  `pip install -U robotframework-browser`
2.  `rfbrowser clean-node`
3.  `rfbrowser init`

## Uninstalling the Library

1.  `rfbrowser clean-node`
2.  `pip uninstall robotframework-browser`

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions.

## Core Team

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

This project thrives on community contributions!  A huge thank you to all contributors for their hard work and dedication.  Supported by [Robocorp](https://robocorp.com/) through [Robot Framework Foundation](https://robotframework.org/foundation/).