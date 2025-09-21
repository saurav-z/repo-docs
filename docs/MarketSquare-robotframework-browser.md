# Robot Framework Browser Library

**Automate your browser testing with speed, reliability, and unparalleled visibility using the Robot Framework Browser library.**  Explore the [original repository](https://github.com/MarketSquare/robotframework-browser) to get started!

[![Version](https://img.shields.io/pypi/v/robotframework-browser.svg)](https://pypi.python.org/pypi/robotframework-browser)
[![Actions Status](https://github.com/MarketSquare/robotframework-browser/workflows/Continuous%20integration/badge.svg)](https://github.com/MarketSquare/robotframework-browser/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Key Features

*   **Powered by Playwright:** Leverage the power of Playwright for fast and reliable browser automation.
*   **Comprehensive Keyword Library:** Access a rich set of keywords for all your browser automation needs. Explore the [keyword documentation](https://marketsquare.github.io/robotframework-browser/Browser.html).
*   **Ergonomic Selectors:** Simplify element selection with a flexible syntax, supporting chaining of `text`, `css`, and `xpath` selectors.
*   **JavaScript Integration:** Extend your tests with custom JavaScript code for advanced automation scenarios.
*   **Asynchronous Operations:** Easily handle asynchronous operations like waiting for HTTP requests and responses.
*   **Device Descriptors:** Test your web applications on various devices using device descriptors.
*   **HTTP Request Handling:** Send HTTP requests and parse their responses directly within your tests.
*   **Parallel Test Execution:** Utilize Pabot for efficient parallel test execution.
*   **Reusable Authentication:** Save and reuse authentication credentials for streamlined testing.

## Installation

### Prerequisites

*   Python 3.9 or newer
*   Node.js (LTS versions 20, 22, and 24 are supported)

### Steps

1.  **Install Node.js:** Download from [https://nodejs.org/en/download/](https://nodejs.org/en/download/)
2.  **Update pip:** `pip install -U pip`
3.  **Install the library:** `pip install robotframework-browser`
4.  **Initialize node dependencies:** `rfbrowser init`
    *   If `rfbrowser` is not found, try `python -m Browser.entry init`
    *   Install Chromium, Firefox and WebKit browsers. Installation size will be at least 700MB.
        *   To skip browser binaries installation: `rfbrowser init --skip-browsers`
        *   To install specific browser binaries: `rfbrowser init firefox chromium`

### Using Docker

Utilize pre-built Docker images available at [https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable](https://github.com/MarketSquare/robotframework-browser/pkgs/container/robotframework-browser%2Frfbrowser-stable). Documentation is available at [docker/README.md](https://github.com/MarketSquare/robotframework-browser/blob/main/docker/README.md).

### Install with transformer
To install library with Robotidy, run install with:
`pip install robotframework-browser[tidy]`
Run external Robotidy [transformer](https://robotidy.readthedocs.io/en/stable/external_transformers.html) with command: `rfbrowser transform --transformer-name /path/to/tests`
Example:
`rfbrowser transform --wait-until-network-is-idle /path/to/tests` would transform deprecated `Wait Until Network Is Idle`
keyword to `Wait For Load State` keyword.
To see full list of transformers provided by Browser library, run
command: `rfbrowser transform --help`.

## Update Instructions

1.  **Update the library:** `pip install -U robotframework-browser`
2.  **Clean node dependencies:** `rfbrowser clean-node`
3.  **Install node dependencies for the new version:** `rfbrowser init`

## Uninstall Instructions

1.  **Clean node dependencies and browser binaries:** `rfbrowser clean-node`
2.  **Uninstall the library:** `pip uninstall robotframework-browser`

## Examples

### Robot Framework Example

```RobotFramework
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

### JavaScript Extension Example

```JavaScript
async function myGoToKeyword(url, page, logger) {
    logger("Going to " + url)
    return await page.goto(url);
}
myGoToKeyword.rfdoc = "This is my own go to keyword";
exports.__esModule = true;
exports.myGoToKeyword = myGoToKeyword;
```

```RobotFramework
*** Settings ***
Library   Browser  jsextension=${CURDIR}/mymodule.js

*** Test Cases ***
Example Test
   New Page
   myGoToKeyword   https://www.robotframework.org
```

More examples can be found at [https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015](https://github.com/MarketSquare/robotframework-browser/tree/main/docs/examples/babelES2015) and ready-made extensions at [robotframework-browser-extensions](https://github.com/MarketSquare/robotframework-browser-extensions).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.

## Core Team

*   Mikko Korpela
*   Tatu Aalto
*   Janne Härkönen (Alumnus)
*   Kerkko Pelttari
*   René Rohner

## Contributors

This project thrives thanks to the contributions of its community.  See the full list of contributors in the original README.

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mkorpela"><img src="https://avatars1.githubusercontent.com/u/136885?v=4?s=100" width="100px;" alt="Mikko Korpela"/><br /><sub><b>Mikko Korpela</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=mkorpela" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aaltat"><img src="https://avatars0.githubusercontent.com/u/2665023?v=4?s=100" width="100px;" alt="Tatu Aalto"/><br /><sub><b>Tatu Aalto</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=aaltat" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://robocorp.com"><img src="https://avatars1.githubusercontent.com/u/8512727?v=4?s=100" width="100px;" alt="Antti Karjalainen"/><br /><sub><b>Antti Karjalainen</b></sub></a><br /><a href="#fundingFinding-aikarjal" title="Funding Finding">🔍</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/ismoaro/"><img src="https://avatars2.githubusercontent.com/u/1047173?v=4?s=100" width="100px;" alt="Ismo Aro"/><br /><sub><b>Ismo Aro</b></sub></a><br /><a href="#fundingFinding-IsNoGood" title="Funding Finding">🔍</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://twitter.com/janneharkonen"><img src="https://avatars3.githubusercontent.com/u/159146?v=4?s=100" width="100px;" alt="Janne Härkönen"/><br /><sub><b>Janne Härkönen</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=yanne" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://xylix.fi"><img src="https://avatars1.githubusercontent.com/u/13387304?v=4?s=100" width="100px;" alt="Kerkko Pelttari"/><br /><sub><b>Kerkko Pelttari</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=xylix" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://robocorp.com"><img src="https://avatars3.githubusercontent.com/u/54288445?v=4?s=100" width="100px;" alt="Robocorp"/><br /><sub><b>Robocorp</b></sub></a><br /><a href="#financial-robocorp" title="Financial">💵</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Snooz82"><img src="https://avatars0.githubusercontent.com/u/41592183?v=4?s=100" width="100px;" alt="René"/><br /><sub><b>René</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=Snooz82" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://wordpress.com/read/feeds/39696435"><img src="https://avatars0.githubusercontent.com/u/1123938?v=4?s=100" width="100px;" alt="Bryan Oakley"/><br /><sub><b>Bryan Oakley</b></sub></a><br /><a href="#ideas-boakley" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/idxn"><img src="https://avatars3.githubusercontent.com/u/2438992?v=4?s=100" width="100px;" alt="Tanakiat Srisaranyakul"/><br /><sub><b>Tanakiat Srisaranyakul</b></sub></a><br /><a href="#ideas-idxn" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://visible-quality.blogspot.com"><img src="https://avatars1.githubusercontent.com/u/5338157?v=4?s=100" width="100px;" alt="Maaret Pyhäjärvi"/><br /><sub><b>Maaret Pyhäjärvi</b></sub></a><br /><a href="#userTesting-maaretp" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.tentamen.eu"><img src="https://avatars2.githubusercontent.com/u/777520?v=4?s=100" width="100px;" alt="Karlo Smid"/><br /><sub><b>Karlo Smid</b></sub></a><br /><a href="#userTesting-karlosmid" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aspargillus"><img src="https://avatars0.githubusercontent.com/u/4592889?v=4?s=100" width="100px;" alt="Frank Schimmel"/><br /><sub><b>Frank Schimmel</b></sub></a><br /><a href="#userTesting-Aspargillus" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tuxmux28"><img src="https://avatars3.githubusercontent.com/u/2794048?v=4?s=100" width="100px;" alt="Christoph"/><br /><sub><b>Christoph</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=tuxmux28" title="Tests">⚠️</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mikahanninen"><img src="https://avatars2.githubusercontent.com/u/1019528?v=4?s=100" width="100px;" alt="Mika Hänninen"/><br /><sub><b>Mika Hänninen</b></sub></a><br /><a href="#question-mikahanninen" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.imbus.de"><img src="https://avatars0.githubusercontent.com/u/67375753?v=4?s=100" width="100px;" alt="imbus"/><br /><sub><b>imbus</b></sub></a><br /><a href="#financial-imbus" title="Financial">💵</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Finalrykku"><img src="https://avatars0.githubusercontent.com/u/19802569?v=4?s=100" width="100px;" alt="Niklas"/><br /><sub><b>Niklas</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=Finalrykku" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gdroes"><img src="https://avatars1.githubusercontent.com/u/6716450?v=4?s=100" width="100px;" alt="gdroes"/><br /><sub><b>gdroes</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=gdroes" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://reaktor.com"><img src="https://avatars2.githubusercontent.com/u/71799?v=4?s=100" width="100px;" alt="Reaktor"/><br /><sub><b>Reaktor</b></sub></a><br /><a href="#financial-reaktor" title="Financial">💵</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/adrianyorke"><img src="https://avatars1.githubusercontent.com/u/30093433?v=4?s=100" width="100px;" alt="Adrian Yorke"/><br /><sub><b>Adrian Yorke</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=adrianyorke" title="Documentation">📖</a> <a href="https://github.com/MarketSquare/robotframework-browser/pulls?q=is%3Apr+reviewed-by%3Aadrianyorke" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/wangzimeiyingtao"><img src="https://avatars0.githubusercontent.com/u/70925596?v=4?s=100" width="100px;" alt="Nanakawa"/><br /><sub><b>Nanakawa</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=wangzimeiyingtao" title="Tests">⚠️</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/emanlove"><img src="https://avatars1.githubusercontent.com/u/993527?v=4?s=100" width="100px;" alt="Ed Manlove"/><br /><sub><b>Ed Manlove</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=emanlove" title="Documentation">📖</a> <a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Aemanlove" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/estimation"><img src="https://avatars1.githubusercontent.com/u/16793171?v=4?s=100" width="100px;" alt="Brian Tsao"/><br /><sub><b>Brian Tsao</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Aestimation" title="Bug reports">🐛</a> <a href="#userTesting-estimation" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mawentao119"><img src="https://avatars0.githubusercontent.com/u/26617186?v=4?s=100" width="100px;" alt="charis"/><br /><sub><b>charis</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=mawentao119" title="Code">💻</a> <a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Amawentao119" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/s-galante"><img src="https://avatars2.githubusercontent.com/u/4580052?v=4?s=100" width="100px;" alt="s-galante"/><br /><sub><b>s-galante</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3As-galante" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.elabit.de"><img src="https://avatars3.githubusercontent.com/u/1897410?v=4?s=100" width="100px;" alt="Simon Meggle"/><br /><sub><b>Simon Meggle</b></sub></a><br /><a href="#userTesting-simonmeggle" title="User Testing">📓</a> <a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Asimonmeggle" title="Bug reports">🐛</a> <a href="https://github.com/MarketSquare/robotframework-browser/commits?author=simonmeggle" title="Tests">⚠️</a> <a href="#ideas-simonmeggle" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Anna-Gunda"><img src="https://avatars3.githubusercontent.com/u/13298792?v=4?s=100" width="100px;" alt="Anna-Gunda"/><br /><sub><b>Anna-Gunda</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3AAnna-Gunda" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/anton264"><img src="https://avatars0.githubusercontent.com/u/10194266?v=4?s=100" width="100px;" alt="anton264"/><br /><sub><b>anton264</b></sub></a><br /><a href="#userTesting-anton264" title="User Testing">📓</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/emakaay"><img src="https://avatars.githubusercontent.com/u/72747481?v=4?s=100" width="100px;" alt="emakaay"/><br /><sub><b>emakaay</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Aemakaay" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://virvatuli.itch.io/"><img src="https://avatars.githubusercontent.com/u/29060467?v=4?s=100" width="100px;" alt="Nea Ohvo"/><br /><sub><b>Nea Ohvo</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3AVirvatuli" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/leeuwe"><img src="https://avatars.githubusercontent.com/u/66635066?v=4?s=100" width="100px;" alt="Elout van Leeuwen"/><br /><sub><b>Elout van Leeuwen</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=leeuwe" title="Documentation">📖</a> <a href="#ideas-leeuwe" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/MarketSquare/robotframework-browser/commits?author=leeuwe" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/LDerikx"><img src="https://avatars.githubusercontent.com/u/26576024?v=4?s=100" width="100px;" alt="LDerikx"/><br /><sub><b>LDerikx</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=LDerikx" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/olga-"><img src="https://avatars.githubusercontent.com/u/9334057?v=4?s=100" width="100px;" alt="olga-"/><br /><sub><b>olga-</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=olga-" title="Documentation">📖</a> <a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Aolga-" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bollwyvl"><img src="https://avatars.githubusercontent.com/u/45380?v=4?s=100" width="100px;" alt="Nicholas Bollweg"/><br /><sub><b>Nicholas Bollweg</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=bollwyvl" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://villesalonen.fi"><img src="https://avatars.githubusercontent.com/u/1070813?v=4?s=100" width="100px;" alt="Ville Salonen"/><br /><sub><b>Ville Salonen</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3AVilleSalonen" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://rasjani.github.io"><img src="https://avatars.githubusercontent.com/u/27887?v=4?s=100" width="100px;" alt="Jani Mikkonen"/><br /><sub><b>Jani Mikkonen</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Arasjani" title="Bug reports">🐛</a> <a href="https://github.com/MarketSquare/robotframework-browser/commits?author=rasjani" title="Documentation">📖</a> <a href="#ideas-rasjani" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/MarketSquare/robotframework-browser/commits?author=rasjani" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/JaPyR"><img src="https://avatars.githubusercontent.com/u/7773301?v=4?s=100" width="100px;" alt="Aleh Borysiewicz"/><br /><sub><b>Aleh Borysiewicz</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3AJaPyR" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.binary-overflow.de"><img src="https://avatars.githubusercontent.com/u/25060709?v=4?s=100" width="100px;" alt="Jürgen Knauth"/><br /><sub><b>Jürgen Knauth</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Ajkpubsrc" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dalaakso"><img src="https://avatars.githubusercontent.com/u/50731554?v=4?s=100" width="100px;" alt="dalaakso"/><br /><sub><b>dalaakso</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Adalaakso" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/msirkka"><img src="https://avatars.githubusercontent.com/u/84907426?v=4?s=100" width="100px;" alt="msirkka"/><br /><sub><b>msirkka</b></sub></a><br /><a href="#ideas-msirkka" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/osrjv"><img src="https://avatars.githubusercontent.com/u/29481017?v=4?s=100" width="100px;" alt="Ossi R."/><br /><sub><b>Ossi R.</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=osrjv" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/adrian-evo"><img src="https://avatars1.githubusercontent.com/u/19324942?v=4?s=100" width="100px;" alt="Adrian V."/><br /><sub><b>Adrian V.</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=adrian-evo" title="Code">💻</a> <a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Aadrian-evo" title="Bug reports">🐛</a> <a href="#ideas-adrian-evo" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ssallmen"><img src="https://avatars.githubusercontent.com/u/39527407?v=4?s=100" width="100px;" alt="Sami Sallmén"/><br /><sub><b>Sami Sallmén</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Assallmen" title="Bug reports">🐛</a> <a href="https://github.com/MarketSquare/robotframework-browser/commits?author=ssallmen" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://eliga.fi"><img src="https://avatars.githubusercontent.com/u/114985?v=4?s=100" width="100px;" alt="Pekka Klärck"/><br /><sub><b>Pekka Klärck</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/commits?author=pekkaklarck" title="Code">💻</a> <a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Apekkaklarck" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/janipalsamaki"><img src="https://avatars.githubusercontent.com/u/1157184?v=4?s=100" width="100px;" alt="Jani Palsamäki"/><br /><sub><b>Jani Palsamäki</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Ajanipalsamaki" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AllanMedeiros"><img src="https://avatars.githubusercontent.com/u/34678196?v=4?s=100" width="100px;" alt="AllanMedeiros"/><br /><sub><b>AllanMedeiros</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3AAllanMedeiros" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ealap"><img src="https://avatars1.githubusercontent.com/u/15620712?v=4?s=100" width="100px;" alt="Emmanuel Alap"/><br /><sub><b>Emmanuel Alap</b></sub></a><br /><a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3Aealap" title="Bug reports">🐛</a> <a href="https://github.com/MarketSquare/robotframework-browser/commits?author=ealap" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ankurbhalla-gmail"><img src="https://avatars.githubusercontent.com/u/90744440?v=4?s=100" width="100px;" alt="ankurbhalla-gmail"/><br /><sub><b>ankurbhalla-gmail</b></sub></a><br /><a href="#ideas-ankurbhalla-gmail" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/UliSei"><img src="https://avatars.githubusercontent.com/u/89480399?v=4?s=100" width="100px;" alt="UliSei"/><br /><sub><b>UliSei</b></sub></a><br /><a href="#ideas-UliSei" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/MarketSquare/robotframework-browser/issues?q=author%3AUliSei" title="Bug reports">