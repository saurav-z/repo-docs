# curl_cffi: Python Library for Advanced HTTP Requests

**Bypass bot detection and scrape websites effortlessly with `curl_cffi`, the powerful Python library that lets you impersonate browsers and customize your HTTP requests!**  [View the Original Repo](https://github.com/lexiforest/curl_cffi)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

`curl_cffi` provides a high-performance and flexible Python interface to the `curl-impersonate` fork, allowing you to mimic browser behavior and avoid bot detection. This makes it ideal for web scraping, API interaction, and other tasks where sophisticated HTTP requests are needed.

**Key Features:**

*   ✅ **Browser Impersonation:**  Mimics TLS/JA3 and HTTP/2 fingerprints of various browsers (Chrome, Safari, Firefox, etc.) and their versions.
*   🚀 **High Performance:** Significantly faster than `requests` and `httpx`, matching the speed of `aiohttp` and `pycurl`.  See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   🐍 **Familiar API:**  Offers a user-friendly API similar to `requests`, making it easy to learn and use.
*   📦 **Pre-compiled:** No need to compile on your machine; ready to use out-of-the-box.
*   🔄 **Asynchronous Support:**  Includes `asyncio` support with proxy rotation for concurrent requests.
*   🌐 **HTTP/2 & HTTP/3 Support:** Supports modern HTTP protocols, unlike some other libraries.
*   🔗 **WebSocket Support:** Offers both synchronous and asynchronous WebSocket capabilities.
*   🛡️ **MIT Licensed:**  Use and distribute freely.

##  Why Choose curl_cffi?

`curl_cffi` is a powerful and versatile tool for anyone needing to make sophisticated HTTP requests in Python.  Whether you're a web scraper, an API developer, or just need a robust HTTP client, `curl_cffi` provides the performance, flexibility, and browser impersonation capabilities you need to succeed.

## Install

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

For unstable versions from GitHub:

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

**macOS Dependencies:**

```bash
brew install zstd nghttp2
```

## Usage

### requests-like

```python
import curl_cffi

# Basic GET request with Chrome impersonation
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Example to use a socks proxy
proxies = {"https": "socks://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)

# To specify a browser version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Use a session to persist cookies across multiple requests
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
r = s.get("https://httpbin.org/cookies")
print(r.json())
```

### Asyncio

```python
import asyncio
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

### WebSockets

```python
from curl_cffi import WebSocket

def on_message(ws: WebSocket, message: str | bytes):
    print(message)

ws = WebSocket(on_message=on_message)
ws.run_forever("wss://api.gemini.com/v1/marketdata/BTCUSD")
```

### Asyncio WebSockets

```python
import asyncio
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    ws = await s.ws_connect("wss://echo.websocket.org")
    await asyncio.gather(*[ws.send_str("Hello, World!") for _ in range(10)])
    async for message in ws:
        print(message)
```

## Supported Browsers

`curl_cffi` supports a wide range of browser versions.  See the [docs on impersonation](https://curl-cffi.readthedocs.io/en/latest/impersonate.html) for the latest information and a complete list.

## Ecosystem Integrations

*   **Scrapy:** `scrapy-curl-cffi`, `scrapy-impersonate`, `scrapy-fingerprint`
*   **Requests Integration:**  `curl-adapter`
*   **Httpx Integration:** `httpx-curl-cffi`
*   **Captcha Solvers:** CapSolver, YesCaptcha (See the original README for promo codes)

## Acknowledgements

*   Forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi)
*   Headers/Cookies files are from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py)
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).