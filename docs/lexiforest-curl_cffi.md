# curl_cffi: The Ultimate Python Library for Web Scraping and Impersonation

**Bypass website restrictions and access the web with ease using `curl_cffi`, a powerful Python library that lets you impersonate browsers and utilize cutting-edge features.** ([View the Original Repo](https://github.com/lexiforest/curl_cffi))

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` is a Python binding for `curl-impersonate`, offering advanced capabilities for web scraping, bypassing Cloudflare, and more. It leverages the power of [cffi](https://cffi.readthedocs.io/en/latest/) to provide a robust and efficient solution. For commercial support, visit [impersonate.pro](https://impersonate.pro).

## Key Features

*   **Browser Impersonation:** Mimic the behavior of various browsers (Chrome, Safari, Firefox, and more) to bypass anti-bot measures using TLS/JA3 and HTTP/2 fingerprints.
*   **High Performance:** Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl`, as shown in our [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Requests-Like API:** Easy to learn and use, mirroring the familiar `requests` API.
*   **Pre-compiled & Cross-Platform:** No need to compile on your machine, works out of the box on Linux, macOS, and Windows.
*   **Asynchronous Support:** Built-in `asyncio` integration with proxy rotation for efficient, concurrent requests.
*   **HTTP/2 & HTTP/3 Support:** Supports the latest web protocols, which are not available in `requests`.
*   **WebSockets:** Offers comprehensive WebSocket support for real-time data retrieval.
*   **Comprehensive Browser Fingerprint Database:** Benefit from a wide range of supported browser versions and configurations.

## Feature Comparison

| Feature         | requests | aiohttp | httpx | pycurl | curl_cffi |
| --------------- | -------- | ------- | ----- | ------ | --------- |
| HTTP/2          | ❌       | ❌      | ✅    | ✅     | ✅        |
| HTTP/3          | ❌       | ❌      | ❌    | ☑️<sup>1</sup>| ✅<sup>2</sup>   |
| Sync            | ✅       | ❌      | ✅    | ✅     | ✅        |
| Async           | ❌       | ✅      | ✅    | ❌     | ✅        |
| WebSocket       | ❌       | ✅      | ❌    | ❌     | ✅        |
| Fingerprints    | ❌       | ❌      | ❌    | ❌     | ✅        |
| Speed           | 🐇       | 🐇🐇     | 🐇    | 🐇🐇    | 🐇🐇       |

**Notes:**
1.  For pycurl, you need an HTTP/3 enabled libcurl.
2.  Available since v0.11.4.

## Installation

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

For unstable versions (from GitHub):

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

On macOS, install the following dependencies:

```bash
brew install zstd nghttp2
```

## Usage

`curl_cffi` provides both a low-level `curl` API and a high-level `requests`-like API.

### Requests-like API

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())
# -> {..., "ja3n_hash": "...", ...}

# Use the latest version of Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")

# Use a specific Chrome version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Use proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
# -> <Cookies[<Cookie foo=bar for httpbin.org />]>

r = s.get("https://httpbin.org/cookies")
print(r.json())
# -> {'cookies': {'foo': 'bar'}}
```

### Supported Browsers

`curl_cffi` supports a wide range of browser versions as detailed in the original README. Access comprehensive browser fingerprints and configurations, including Chrome, Safari, Firefox, and more.

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

## Ecosystem

*   **Scrapy Integration:** `divtiply/scrapy-curl-cffi`, `jxlil/scrapy-impersonate`, and `tieyongjie/scrapy-fingerprint`.
*   **Adapters:**  Integrate with `requests` (`el1s7/curl-adapter`) and `httpx` (`vgavro/httpx-curl-cffi`).
*   **Captcha Resolvers:**  Supports integration with  `CapSolver` and `YesCaptcha`.

## Sponsors

This project is supported by the contributions of many people. If you'd like to sponsor this project and have your avatar or company logo appear below <a href="https://github.com/sponsors/lexiforest">click here</a>. 💖

## Bypass Cloudflare with API

<a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>

Yescaptcha is a proxy service that bypasses Cloudflare and uses the API interface to obtain verified cookies (e.g. `cf_clearance`). Click [here](https://yescaptcha.com/i/stfnIO) to register: https://yescaptcha.com/i/stfnIO

## Acknowledgements

*   Forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi) (MIT License).
*   Headers/Cookies files from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py) (BSD License).
*   Asyncio support inspired by Tornado's curl http client.
*   Synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   Asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).

## Contributing

We welcome contributions! Please submit pull requests using a branch other than `main` and check the "Allow edits by maintainers" box.