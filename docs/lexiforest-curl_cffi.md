# curl_cffi: Python Binding for Advanced Web Scraping & Impersonation

**Bypass bot detection and effortlessly scrape websites with `curl_cffi`, the leading Python library for emulating browser behavior.** ([Original Repo](https://github.com/lexiforest/curl_cffi))

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)
[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` is a powerful Python library built upon a [curl-impersonate fork](https://github.com/lexiforest/curl-impersonate) and  [cffi](https://cffi.readthedocs.io/en/latest/). It allows you to mimic browser fingerprints for advanced web scraping and bypassing bot detection. Get commercial support from [impersonate.pro](https://impersonate.pro).

## Key Features

*   **Browser Impersonation:**  Mimic TLS/JA3 and HTTP/2 fingerprints of popular browsers (Chrome, Safari, Firefox, Edge, etc.) and even custom fingerprints.
*   **High Performance:** Significantly faster than `requests` and `httpx`, and comparable to `aiohttp/pycurl`, as shown in [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:**  Uses a  `requests`-like API for easy adoption.
*   **Pre-compiled Binaries:** No need to compile on your machine, streamlining setup.
*   **Asynchronous Support:**  Includes `asyncio` support with proxy rotation for efficient concurrent requests.
*   **HTTP/2 & HTTP/3 Support:**  Supports both HTTP/2 and HTTP/3, giving you the latest protocol support.
*   **WebSockets:** Built-in support for WebSockets for real-time data streaming.
*   **MIT License:**  Use and integrate freely in your projects.

## Installation

Install `curl_cffi` using pip:

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

**Dependencies for macOS:**

```bash
brew install zstd nghttp2
```

## Usage

`curl_cffi` offers a flexible API including a `requests`-like and a low-level `curl` API.

### Requests-like API (Simplified)

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Pin specific browser versions
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Proxies supported
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
```

### Supported Browsers

`curl_cffi` supports the following browsers and versions:

| Browser         | Open Source                                                                                                                                                                                             | Pro version (Commercial)                                                                                               |
| :-------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| Chrome          | chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116, chrome119, chrome120, chrome123, chrome124, chrome131, chrome133a, chrome136                                                  | chrome132, chrome134, chrome135                                                                                       |
| Chrome Android  | chrome99_android, chrome131_android                                                                                                                                                                    | chrome132_android, chrome133_android, chrome134_android, chrome135_android                                            |
| Chrome iOS      | N/A                                                                                                                                                                                                    | coming soon                                                                                                            |
| Safari          | safari153, safari155, safari170, safari180, safari184, safari260                                                                                                                                    | coming soon                                                                                                            |
| Safari iOS      | safari172_ios, safari180_ios, safari184_ios, safari260_ios                                                                                                                                          | coming soon                                                                                                            |
| Firefox         | firefox133, firefox135                                                                                                                                                                                  | coming soon                                                                                                            |
| Firefox Android | N/A                                                                                                                                                                                                    | firefox135_android                                                                                                     |
| Tor             | tor145                                                                                                                                                                                                   | coming soon                                                                                                            |
| Edge            | edge99, edge101                                                                                                                                                                                          | edge133, edge135                                                                                                       |
| Opera           | N/A                                                                                                                                                                                                    | coming soon                                                                                                            |
| Brave           | N/A                                                                                                                                                                                                    | coming soon                                                                                                            |

**Note:** Consider purchasing commercial support from [impersonate.pro](https://impersonate.pro) for comprehensive browser fingerprints database.

### Asynchronous Usage

```python
from curl_cffi import AsyncSession
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

## Ecosystem Integrations

*   **Scrapy:**  `scrapy-curl-cffi`, `scrapy-impersonate`, `scrapy-fingerprint`
*   **Requests & httpx Adapters:** `curl-adapter`, `httpx-curl-cffi`
*   **Captcha Resolvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview)

## Acknowledgements

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
*   Headers/Cookies files are from [httpx](https://github.com/encode/httpx).
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).