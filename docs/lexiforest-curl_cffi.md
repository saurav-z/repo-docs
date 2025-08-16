# curl-cffi: The Fastest Python HTTP Client with Browser Impersonation 🚀

**Bypass website restrictions and achieve superior performance with `curl-cffi`, the Python library that lets you effortlessly impersonate browsers for robust web scraping and API interaction.**  Check out the original repo [here](https://github.com/lexiforest/curl_cffi).

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

📖 [Documentation](https://curl-cffi.readthedocs.io) | 🏢 [Commercial Support](https://impersonate.pro)

`curl-cffi` is a high-performance Python binding for `curl` that leverages the `curl-impersonate` fork to mimic browser behavior, making it ideal for bypassing bot detection and accessing websites that restrict access. It outperforms traditional HTTP clients like `requests` and `httpx`, offering speed and advanced features.

## Key Features

*   **Browser Impersonation:**  Seamlessly impersonate popular browsers (Chrome, Safari, Firefox, Edge, Tor, etc.) and their various versions, including JA3/TLS and HTTP/2 fingerprints.
*   **Blazing Fast Performance:**  Outperforms `requests`, `httpx`, and often rivals `aiohttp/pycurl` in speed. See the [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API, minimizing the learning curve.
*   **Pre-compiled:**  No need to compile on your machine.
*   **Asyncio Support:**  Built-in support for `asyncio`, including proxy rotation.
*   **HTTP/2 & HTTP/3 Support:**  Supports both HTTP/2 and HTTP/3, features missing from the requests library.
*   **WebSocket Support:**  Includes synchronous and asynchronous WebSocket capabilities.
*   **MIT License:**  Free to use and modify.

## Speed Comparison

| Feature         | requests | aiohttp | httpx | pycurl | curl-cffi |
|-----------------|----------|---------|-------|--------|-----------|
| HTTP/2          | ❌        | ❌       | ✅    | ✅     | ✅         |
| HTTP/3          | ❌        | ❌       | ❌    | ☑️<sup>1</sup> | ✅<sup>2</sup>   |
| Sync            | ✅        | ❌       | ✅    | ✅     | ✅         |
| Async           | ❌        | ✅       | ✅    | ❌     | ✅         |
| WebSocket       | ❌        | ✅       | ❌    | ❌     | ✅         |
| Fingerprints    | ❌        | ❌       | ❌    | ❌     | ✅         |
| Speed           | 🐇        | 🐇🐇     | 🐇   | 🐇🐇   | 🐇🐇       |

Notes:
1.  For pycurl, you need an http/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2.  Since v0.11.4.

## Installation

Install `curl-cffi` with pip:

```bash
pip install curl_cffi --upgrade
```

This will work on Linux, macOS, and Windows. If issues arise, manual compilation of `curl-impersonate` might be necessary, along with setting environment variables like `LD_LIBRARY_PATH`.

*   Install Beta Releases: `pip install curl_cffi --upgrade --pre`
*   Install from GitHub:

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

On macOS, install dependencies:

```bash
brew install zstd nghttp2
```

## Usage Examples

### Requests-like API

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Pin a specific version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Using proxies
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

### Supported Browsers

`curl-cffi` supports various browser versions.  For more detailed information on browser versions and commercial support, see the [docs](https://curl-cffi.readthedocs.io/en/latest/impersonate.html) or visit [impersonate.pro](https://impersonate.pro).

### Asyncio Example

```python
from curl_cffi import AsyncSession
import asyncio

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

## Ecosystem Integration

*   **Scrapy:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Requests Adapter:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter)
*   **HTTX Adapter:** [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi)
*   **Captcha Resolvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Acknowledgements

*   Forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
*   Headers/Cookies files from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py).
*   Asyncio support inspired by Tornado's curl.
*   Synchronous WebSocket API inspired by [websocket\_client](https://github.com/websocket-client/websocket-client).
*   Asynchronous WebSocket API inspired by [aiohttp](https://github.com/aio-libs/aiohttp).

## Contributing

Contributions are welcome! Please create a branch other than `main` for your pull requests and check the "Allow edits by maintainers" box.