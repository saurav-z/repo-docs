# Robot Framework Browser Library: Automate Web Testing with Ease

🤖 Elevate your web automation with the Robot Framework Browser library, powered by Playwright, offering speed, reliability, and in-depth visibility.  [Check it out on GitHub](https://github.com/MarketSquare/robotframework-browser)

**Key Features:**

*   🚀 **Blazing Fast:** Experience rapid automation through Playwright.
*   ✅ **Rock-Solid Reliability:**  Ensure dependable test execution.
*   🔬 **Enhanced Visibility:** Gain deep insights into your test runs.
*   ⚙️ **Ergonomic Selectors:**  Use intuitive CSS, XPath, and text selectors.
*   🔌 **JavaScript Extension:**  Extend functionality with custom JavaScript.
*   📱 **Device Emulation:** Test on various devices using device descriptors.
*   📡 **HTTP Request Handling:** Send and parse HTTP requests and responses.
*   🚦 **Asynchronous Operations:**  Handle asynchronous requests and responses.
*   📦 **Built-in Transformers:** Convert deprecated keywords automatically (e.g., `Wait Until Network Is Idle` to `Wait For Load State`).

## Installation

1.  **Prerequisites:** Ensure you have Python 3.9 or newer and Node.js (versions 20 and 22 LTS supported) installed.  Download from [https://nodejs.org/en/download/](https://nodejs.org/en/download/).
2.  **Update Pip:**  `pip install -U pip`
3.  **Install Library:**  `pip install robotframework-browser`
4.  **Initialize Node Dependencies:**  `rfbrowser init`  (or `python -m Browser.entry init` if `rfbrowser` is not found).

    *   By default, the library installs Chromium, Firefox, and WebKit browsers.
    *   Use `rfbrowser init --skip-browsers` to skip browser binary installation, but you'll be responsible for their installation.
    *   Specify browsers to install:  `rfbrowser init firefox chromium`.

## Installation with transformer

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

**Robot Framework:**

```robotframework
*** Settings ***
Library   Browser

*** Test Cases ***
Example Test
    New Page    https://playwright.dev
    Get Text    h1    contains    Playwright
```

**Python:**

```python
import Browser
browser = Browser.Browser()
browser.new_page("https://playwright.dev")
assert 'Playwright' in browser.get_text("h1")
browser.close_browser()
```

**JavaScript Extension:**

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
**Ergonomic Selector Syntax:**
```robotframework
# Select element containing text "Login" with text selector strategy
# and select it's parent `input` element with xpath
Click    "Login" >> xpath=../input
# Select element with CSS strategy and select button in it with text strategy
Click    div.dialog >> "Ok"
```
**Evaluate in browser page:**
```robotframework
New Page   ${LOGIN_URL}
${ref}=    Get Element    h1
Get Property    ${ref}    innerText    ==    Login Page
Evaluate JavaScript    ${ref}    (elem) => elem.innerText = "abc"
Get Property    ${ref}    innerText    ==    abc
```
**Asynchronously waiting for HTTP requests and responses:**
```robotframework
# The button with id `delayed_request` fires a delayed request. We use a promise to capture it.
${promise}=    Promise To    Wait For Response    matcher=    timeout=3s
Click    \#delayed_request
${body}=    Wait For    ${promise}
```
**Device Descriptors:**
```robotframework
${device}=  Get Device  iPhone X
New Context  &{device}
New Page
Get Viewport Size  # returns { "width": 375, "height": 812 }
```
**Sending HTTP requests and parsing their responses:**
```robotframework
${response}=    HTTP    /api/post    POST    {"name": "John"}
Should Be Equal    ${response.status}    ${200}
```

**More Examples:** Explore comprehensive examples in the [example](https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015) directory, and contribute your own at [robotframework-browser-extensions](https://github.com/MarketSquare/robotframework-browser-extensions).

## Parallel Testing with Pabot

*   Use `pabot` to run tests in parallel.
*   Avoid using `--testlevelsplit` for smaller tests to reduce overhead.
*   Share RF Browser node processes with `ROBOT_FRAMEWORK_BROWSER_NODE_PORT` environment variable.
*   Clean up processes afterwards.

## Re-using Authentication Credentials

*   Use `Save Storage State` with local storage or cookies.  See example at [https://marketsquare.github.io/robotframework-browser/Browser.html#Save%20Storage%20State](https://marketsquare.github.io/robotframework-browser/Browser.html#Save%20Storage%20State)

## Development

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Core Team

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

Thank you to all [187+ contributors](https://github.com/MarketSquare/robotframework-browser/graphs/contributors) who have made this project possible!