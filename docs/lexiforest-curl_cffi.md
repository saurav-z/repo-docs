# curl_cffi: Mimic Browser Fingerprints with Python & libcurl

**Bypass website restrictions and scrape data effectively with curl_cffi, a powerful Python library that mimics browser fingerprints to avoid detection.**  [See the original repo](https://github.com/lexiforest/curl_cffi)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Generic badge](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Generic badge](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` is a Python binding for the [curl-impersonate fork](https://github.com/lexiforest/curl-impersonate) via [cffi](https://cffi.readthedocs.io/en/latest/), designed to bypass bot detection and web scraping restrictions. It offers browser fingerprinting capabilities, making it a robust alternative to traditional HTTP clients like `requests` and `httpx`. For commercial support, visit [impersonate.pro](https://impersonate.pro).

## Key Features

*   **Browser Impersonation:** Emulates various browser fingerprints (TLS/JA3 and HTTP/2) including Chrome, Safari, Firefox, and more, for advanced scraping.
*   **High Performance:** Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`, as shown in [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:**  Offers a user-friendly API similar to the `requests` library, minimizing the learning curve.
*   **Asynchronous Support:** Includes full support for `asyncio`, and proxy rotation for asynchronous requests.
*   **HTTP/2 & HTTP/3 Support:** Offers native support for both HTTP/2 and HTTP/3 protocols.
*   **Websocket Support:** Provides websocket functionality.
*   **Cross-Platform:** Pre-compiled wheels available for Linux, macOS, and Windows.

## Installation

```bash
pip install curl_cffi --upgrade
```

## Usage

`curl_cffi` has both a low-level `curl` API and a high-level, `requests`-like API.

### Requests-like Example

```python
import curl_cffi

r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())
```

### Sessions Example

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
r = s.get("https://httpbin.org/cookies")
print(r.json())
```

### Asyncio Example

```python
import asyncio
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

## Supported Impersonated Browsers

`curl_cffi` supports the browser versions available in the linked [curl-impersonate fork](https://github.com/lexiforest/curl-impersonate)
The following table lists some open source options. Refer to the original documentation for the most up-to-date and complete list, and details on `ja3` and `akamai` options for custom fingerprints.

| Browser          | Open Source Versions                                                                                                                                                                                                                             | Pro Versions                                 |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Chrome           | chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116, chrome119, chrome120, chrome123, chrome124, chrome131, chrome133a, chrome136           | chrome132, chrome134, chrome135                 |
| Chrome Android   | chrome99_android, chrome131_android                                                                                                                                                                                                                   | chrome132_android, chrome133_android, chrome134_android, chrome135_android  |
| Safari           | safari153, safari155, safari170, safari180, safari184, safari260                                                                                                                                                                                         | coming soon                                 |
| Safari iOS       | safari172_ios, safari180_ios, safari184_ios, safari260_ios                                                                                                                                                                                           | coming soon                                 |
| Firefox          | firefox133, firefox135                                                                                                                                                                                                                                | coming soon                                 |
| Firefox Android  | N/A                                                                                                                                                                                                                                                   | firefox135_android                           |
| Tor              | tor145                                                                                                                                                                                                                                                  | coming soon                                 |
| Edge             | edge99, edge101                                                                                                                                                                                                                                        | edge133, edge135                           |
| Opera            | N/A                                                                                                                                                                                                                                                   | coming soon                                 |
| Brave            | N/A                                                                                                                                                                                                                                                   | coming soon                                 |

## Ecosystem Integrations

`curl_cffi` integrates with several popular tools and services:

*   **Scrapy:**  [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint)
*   **Adapters:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) for `requests`, and [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) for `httpx`.
*   **Captcha Resolvers:** [CapSolver](https://docs.capsolver.com/en/api/) and [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview)

## Sponsors & Support

Maintenance of this project is made possible by all the [contributors](https://github.com/lexiforest/curl_cffi/graphs/contributors) and [sponsors](https://github.com/sponsors/lexiforest). If you'd like to sponsor this project and have your avatar or company logo appear below [click here](https://github.com/sponsors/lexiforest). 💖

### Recommended Services for Bypass & Automation

Leverage the power of `curl_cffi` in conjunction with these services to simplify your scraping tasks:

1.  **SerpAPI:**

    [<img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/serpapi.png" alt="SerpAPI" height="67" width="63">](https://serpapi.com/)

    Scrape Google and other search engines with SerpApi's fast, reliable API.

2.  **Yescaptcha:**

    [<img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149">](https://yescaptcha.com/i/stfnIO)

    Bypass Cloudflare using Yescaptcha to obtain verified cookies.

3.  **CapSolver:**

    [<img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/capsolver.jpg" alt="CapSolver" height="50" width="178">](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC)

    Easily bypass Captchas with CapSolver, an AI-powered tool. Use code "CURL" for a 6% balance bonus!

## Acknowledgements

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi), MIT licensed.
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py), BSD licensed.
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).