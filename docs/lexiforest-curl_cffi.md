# curl_cffi: The Fastest Python HTTP Client for Impersonating Browsers

Tired of being blocked? **curl_cffi** is a powerful Python library built on `curl-impersonate` that allows you to mimic browser fingerprints, bypass anti-bot defenses, and scrape websites effectively. Visit the [original repo](https://github.com/lexiforest/curl_cffi) for the full source code.

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

## Key Features

*   **Browser Impersonation:** Mimic various browser fingerprints (TLS/JA3, HTTP/2) to bypass bot detection.
*   **High Performance:** Significantly faster than `requests` and `httpx`, on par with `aiohttp` and `pycurl`.
*   **Easy to Use:**  Offers a familiar `requests`-like API, minimizing the learning curve.
*   **Asynchronous Support:** Includes full `asyncio` support with proxy rotation.
*   **HTTP/2 & HTTP/3 Compatibility:**  Supports the latest HTTP versions for improved speed and compatibility.
*   **Websocket Support:** Enables seamless interaction with websocket APIs.
*   **Pre-compiled:**  No need for manual compilation on your machine.

## Core Advantages

*   **Bypass Website Restrictions:** Successfully access websites that block standard HTTP clients.
*   **High-Speed Scraping:** Achieve faster data extraction with superior performance.
*   **Customizable:** Control JA3/Akamai fingerprints to impersonate other clients than browsers.

## Installation

```bash
pip install curl_cffi --upgrade
```
Refer to the original [repo](https://github.com/lexiforest/curl_cffi) for detailed installation instructions, including beta and unstable releases.

## Usage

### Requests-like API

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Pin specific versions:
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Use proxies:
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
```

### Asyncio

```python
import asyncio
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

## Browser Support

`curl_cffi` supports various browser versions. Check the [original README](https://github.com/lexiforest/curl_cffi) for a full list.

## Ecosystem

*   [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi)
*   [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate)
*   [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint)
*   [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter)
*   [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi)
*   [CapSolver](https://docs.capsolver.com/en/api/)
*   [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview)

## Acknowledgements

*   Forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi)
*   Headers/Cookies files are from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py)
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).

---