# curl_cffi: The Ultimate Python HTTP Client for Web Scraping & Impersonation

Tired of getting blocked? **curl_cffi lets you effortlessly mimic browser behavior for seamless web scraping and bypass restrictions.** ([Original Repo](https://github.com/lexiforest/curl_cffi))

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io) | [Commercial Support](https://impersonate.pro)

`curl_cffi` is a powerful Python binding that leverages the `curl-impersonate` fork for advanced web scraping and bypassing anti-bot measures. It's designed to mimic browser fingerprints, enabling you to access websites that may block standard HTTP clients.

## Key Features:

*   **Browser Impersonation:** Effortlessly mimic Chrome, Safari, and other browser fingerprints, including TLS/JA3 and HTTP/2.
*   **High Performance:**  Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl`.
*   **Familiar API:** Uses a `requests`-like API, minimizing the learning curve.
*   **Asyncio Support:** Includes built-in `asyncio` support with proxy rotation for asynchronous requests.
*   **HTTP/2 & HTTP/3 Support:** Includes support for modern protocols.
*   **Websocket Support:** Includes both synchronous and asynchronous Websocket APIs.
*   **Pre-compiled:** No need to compile dependencies on your machine.

## Benchmarks

`curl_cffi`'s speed is comparable to `aiohttp` and `pycurl`. More details can be found in the [benchmark](https://github.com/lexiforest/curl_cffi/tree/main/benchmark) directory.

## Installation

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

## Usage Examples

### Requests-like API

```python
import curl_cffi

r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
r = s.get("https://httpbin.org/cookies")
print(r.json())
```

### Asyncio

```python
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

## Supported Impersonate Browsers

`curl_cffi` supports various browser versions. For an up-to-date list, refer to the original README. Consider checking [impersonate.pro](https://impersonate.pro) for a comprehensive database of browser fingerprints.

## Ecosystem

*   **Scrapy Integration:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapter Integration:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi).
*   **Captcha Resolvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Acknowledgements

This project builds upon the work of others, including:

*   [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi)
*   [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py)
*   Tornado's curl http client
*   [websocket_client](https://github.com/websocket-client/websocket-client)
*   [aiohttp](https://github.com/aio-libs/aiohttp)

## Contributing

We welcome contributions! Please submit pull requests from a branch other than `main` and check the "Allow edits by maintainers" box.