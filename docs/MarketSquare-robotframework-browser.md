# Robot Framework Browser Library: Modern Browser Automation

**Automate your web testing with ease using the Robot Framework Browser library, powered by Playwright!**  [Explore the original repository](https://github.com/MarketSquare/robotframework-browser).

[![Version](https://img.shields.io/pypi/v/robotframework-browser.svg)](https://pypi.python.org/pypi/robotframework-browser)
[![Actions Status](https://github.com/MarketSquare/robotframework-browser/workflows/Continuous%20integration/badge.svg)](https://github.com/MarketSquare/robotframework-browser/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Key Features

*   **Fast and Reliable Automation:** Experience blazing-fast test execution with Playwright.
*   **Comprehensive Keyword Library:** Access a wide range of keywords for all your browser automation needs.
*   **Ergonomic Selectors:** Use intuitive and powerful selector syntax for easy element targeting.
*   **JavaScript Integration:** Extend functionality using JavaScript extensions.
*   **Asynchronous Operations:** Seamlessly handle asynchronous HTTP requests and responses for robust testing.
*   **Device Descriptors:**  Test responsiveness across different devices with ease.
*   **Integration with Robotidy:**  Automate code formatting with the optional Robotidy integration.
*   **Parallel Test Execution:** Easily run tests in parallel with Pabot.
*   **Flexible Authentication Handling:**  Effortlessly reuse authentication credentials for efficient testing.

## Installation

### Prerequisites

*   Node.js (20 and 22 LTS versions supported)
*   Python 3.9 or newer
*   pip (ensure latest version: `pip install -U pip`)

### Steps

1.  **Install Node.js:** Download and install from [https://nodejs.org/en/download/](https://nodejs.org/en/download/)
2.  **Install Robot Framework Browser:** `pip install robotframework-browser`
3.  **Initialize Node Dependencies:** Run `rfbrowser init` to install browser binaries (Chromium, Firefox, and WebKit by default).
    *   Customize installation: `rfbrowser init --skip-browsers` to skip browser binary installation or `rfbrowser init firefox chromium` to install specific browsers.

    *   Alternatively, use [docker images](https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable).

### Installation with Transformer (Robotidy)

1.  Install with Robotidy dependency: `pip install robotframework-browser[tidy]`
2.  Run Robotidy transformer: `rfbrowser transform --transformer-name /path/to/tests` (e.g., `rfbrowser transform --wait-until-network-is-idle /path/to/tests`).
    *  See all available transformers: `rfbrowser transform --help`.

## Updating

1.  Update library: `pip install -U robotframework-browser`
2.  Clean node dependencies: `rfbrowser clean-node`
3.  Re-initialize: `rfbrowser init`

## Uninstalling

1.  Clean node dependencies: `rfbrowser clean-node`
2.  Uninstall: `pip uninstall robotframework-browser`

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

*   Integrate custom JavaScript code into your tests. See [example](https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015) and [robotframework-browser-extensions](https://github.com/MarketSquare/robotframework-browser-extensions) for ready-made extensions.

### Selector Syntax

*   Use a chain of selectors like `Click "Login" >> xpath=../input` and `Click div.dialog >> "Ok"`.

### Evaluate JavaScript

*   Execute JavaScript directly in the browser, using the `Evaluate JavaScript` keyword.

### Asynchronous HTTP Request Handling

*   Use `Promise To Wait For Response` and `Wait For` to handle asynchronous operations.

### Device Descriptors

*   Use `Get Device` and `New Context` to simulate different devices.

### Sending HTTP Requests and Parsing Responses

*   Use `HTTP` keyword to send HTTP requests and easily inspect responses.

### Parallel Execution with Pabot

*   Use environment variable `ROBOT_FRAMEWORK_BROWSER_NODE_PORT` and helper `spawn_node_process` for sharing RF Browser processes.

### Re-using Authentication Credentials

*   Use `Save Storage State` to manage authentication.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.

## Core Team

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

This project is community-driven.  Special thanks to all contributors!