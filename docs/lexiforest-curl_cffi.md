# curl_cffi: Bypass Website Restrictions with Python's Fastest and Most Advanced HTTP Client

**Tired of being blocked?** `curl_cffi` is the ultimate Python library for high-performance web scraping and API interaction, allowing you to seamlessly impersonate browsers and bypass website restrictions.  Visit the original repository: [https://github.com/lexiforest/curl_cffi](https://github.com/lexiforest/curl_cffi).

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

Built upon a `curl-impersonate` fork and `cffi`, `curl_cffi` empowers you to access websites that block standard Python HTTP clients, offering unparalleled speed and flexibility.

## Key Features

*   **Browser Impersonation:**  Seamlessly mimic Chrome, Safari, and other browsers' TLS/JA3 and HTTP/2 fingerprints to bypass bot detection.
*   **Blazing Fast Performance:** Outperforms `requests` and `httpx`, rivaling `aiohttp` and `pycurl` in speed, making it ideal for high-volume scraping.
*   **Familiar API:** Uses a `requests`-like API for easy adoption, minimizing your learning curve.
*   **Pre-compiled:**  Ready to use out-of-the-box on Linux, macOS, and Windows, eliminating the need for manual compilation.
*   **Asyncio Support:**  Includes built-in `asyncio` support with proxy rotation for asynchronous operations.
*   **HTTP/2 & HTTP/3 Support:** Supports HTTP/2 and HTTP/3, which many other Python HTTP clients lack.
*   **Websocket Support:** Added support for Websockets.
*   **Versatile & Flexible:** Supports custom JA3/Akamai fingerprints, proxies (HTTP/SOCKS), and offers both low-level and high-level APIs.
*   **MIT License:**  Free to use and integrate in your projects.

## Features Comparison

| Feature         | requests | aiohttp | httpx | pycurl | curl_cffi |
| --------------- | -------- | ------- | ----- | ------ | --------- |
| HTTP/2          | ❌      | ❌      | ✅    | ✅     | ✅        |
| HTTP/3          | ❌      | ❌      | ❌    | ✅<sup>1</sup>     | ✅<sup>2</sup>       |
| Sync            | ✅      | ❌      | ✅    | ✅     | ✅        |
| Async           | ❌      | ✅      | ✅    | ❌     | ✅        |
| Websocket       | ❌      | ✅      | ❌    | ❌     | ✅        |
| Fingerprints    | ❌      | ❌      | ❌    | ❌     | ✅        |
| Speed           | 🐇      | 🐇🐇     | 🐇    | 🐇🐇    | 🐇🐇       |

**Notes:**
1.  For pycurl, you need an http/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2.  Since v0.11.4.

## Installation

Install `curl_cffi` quickly using pip:

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

## Usage

`curl_cffi` offers both a `requests`-like API and a low-level `curl` API.

### Requests-like API

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())  # Shows the Chrome JA3 fingerprint

# Use a specific Chrome version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Use Real World Browser
r = curl_cffi.get("https://example.com", impersonate="realworld")

# Use other fingerprints
r = curl_cffi.get("https://tls.browserleaks.com/json", ja3="...", akamai="...")

# Use proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")  # Set cookies
print(s.cookies)
r = s.get("https://httpbin.org/cookies")
print(r.json())  # {'cookies': {'foo': 'bar'}}
```

### Supported Browsers & Fingerprints

`curl_cffi` supports a wide range of browser versions, including Chrome, Safari, Firefox, Edge, and more.  See the [docs on impersonation](https://curl-cffi.readthedocs.io/en/latest/impersonate/_index.html) for detailed version support and advanced usage. Commercial support is available at [impersonate.pro](https://impersonate.pro) for comprehensive browser fingerprints.

| Browser           | Open Source Versions                                                                     | Pro Version                                |
| ----------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------ |
| Chrome            | chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116, chrome119, chrome120, chrome123, chrome124, chrome131, chrome133a, chrome136 | chrome132, chrome134, chrome135            |
| Chrome Android    | chrome99_android, chrome131_android                                                       | chrome132_android, chrome133_android, chrome134_android, chrome135_android |
| Chrome iOS        | N/A                                                                                      | coming soon                               |
| Safari            | safari153, safari155, safari170, safari180, safari184, safari260                                                      | coming soon                               |
| Safari iOS        | safari172_ios, safari180_ios, safari184_ios, safari260_ios                                                      | coming soon                               |
| Firefox           | firefox133, firefox135                                                                   | coming soon                               |
| Firefox Android   | N/A                                                                                      | firefox135_android                         |
| Tor               | tor145                                                                                   | coming soon                               |
| Edge              | edge99, edge101                                                                          | edge133, edge135                           |
| Opera             | N/A                                                                                      | coming soon                               |
| Brave             | N/A                                                                                      | coming soon                               |

### Asyncio Example

```python
from curl_cffi import AsyncSession
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

### WebSockets Example

```python
from curl_cffi import WebSocket

def on_message(ws: WebSocket, message: str | bytes):
    print(message)

ws = WebSocket(on_message=on_message)
ws.run_forever("wss://api.gemini.com/v1/marketdata/BTCUSD")
```

## Ecosystem Integrations

*   **Scrapy:**  [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Requests & httpx Adapters:**  [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi).
*   **Captcha Resolvers:**  [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Acknowledgments

*   Original fork from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi), MIT License.
*   Headers/Cookies files from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py), BSD license.
*   Asyncio support inspired by Tornado and aiohttp.
*   Synchronous WebSocket API inspired by websocket_client.

## Contributing

We welcome contributions!  Please use a branch other than `main` for your PR and check the "Allow edits by maintainers" box.