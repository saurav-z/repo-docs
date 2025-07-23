# curl_cffi: Python's Premier Library for Mimicking Browser Behavior

**Bypass web scraping restrictions and enhance your web interaction capabilities with `curl_cffi`, the Python library that allows you to impersonate browser fingerprints.**  [See the original repo](https://github.com/lexiforest/curl_cffi)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` is a powerful Python binding for the `curl-impersonate` fork, built on `cffi`. It offers unparalleled control over your HTTP requests, enabling you to mimic the behavior of modern web browsers.  Ideal for scraping, testing, and bypassing anti-bot measures.

**Key Features:**

*   **Browser Impersonation:**  Seamlessly mimics the TLS/JA3 and HTTP/2 fingerprints of various browsers, including Chrome, Safari, Firefox, and more.  Avoids detection by sophisticated anti-bot systems.
*   **High Performance:** Significantly faster than standard Python HTTP clients like `requests` and `httpx`, rivaling `aiohttp` and `pycurl`.  See the [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:**  Offers a `requests`-like API, making it easy to integrate into existing projects.  No need to learn a completely new library.
*   **Asynchronous Support:** Includes robust `asyncio` support with proxy rotation for efficient, concurrent requests.
*   **HTTP/2 & HTTP/3 Support:**  Built-in support for modern HTTP protocols, including HTTP/2 and HTTP/3.
*   **WebSockets:**  Full support for WebSockets, both synchronous and asynchronous.
*   **Pre-compiled:**  No need to compile on your machine.
*   **Commercial Support:** Visit [impersonate.pro](https://impersonate.pro) for commercial support.

## Installation

```bash
pip install curl_cffi --upgrade
```

**(Optional) Install beta/unstable versions:**
```bash
pip install curl_cffi --upgrade --pre # Beta releases
```
```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install . # Unstable from GitHub
```
*Note:* macOS users may need to install `zstd` and `nghttp2` using `brew`.

## Usage

`curl_cffi` provides both a high-level `requests`-like API and a low-level `curl` API.

### Requests-like API (v0.10 and later)

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Pin a specific version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Use proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()

# set cookies
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)

# verify cookies
r = s.get("https://httpbin.org/cookies")
print(r.json())
```

### Supported Browsers

`curl_cffi` supports a wide range of browser versions, as listed in the full README.  The open-source version provides many versions of Chrome, Firefox, and Safari. Consider commercial support for more comprehensive browser fingerprint databases [impersonate.pro](https://impersonate.pro)

### Asyncio

```python
from curl_cffi import AsyncSession
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com")
    print(r.text)

# More Concurrency
async with AsyncSession() as s:
    tasks = [s.get(url) for url in ["https://google.com/", "https://facebook.com/", "https://twitter.com/"]]
    results = await asyncio.gather(*tasks)
```

### WebSockets

```python
from curl_cffi import WebSocket
import asyncio

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

## Ecosystem

`curl_cffi` integrates seamlessly with other tools to enhance your web interaction capabilities:

*   **Scrapy Integration:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapter Integration:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) and [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi).
*   **Captcha Resolvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Acknowledgements

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
*   Headers/Cookies files are from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py).
*   Asyncio support inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).