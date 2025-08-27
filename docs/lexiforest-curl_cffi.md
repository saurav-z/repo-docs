# curl_cffi: The Fastest Python HTTP Client for Impersonating Browsers

**Bypass website restrictions and achieve lightning-fast web scraping with `curl_cffi`, a powerful Python library built on `curl-impersonate` and `cffi`.**  [Explore the Original Repo](https://github.com/lexiforest/curl_cffi)

[![PyPI Downloads](https://static.pepy.tech/badge/curl-cffi/week)](https://pepy.tech/projects/curl-cffi)
[![Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://img.shields.io/pypi/pyversions/curl_cffi)
[![PyPI Version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi.svg)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` is a robust Python binding for the `curl-impersonate` fork, leveraging the `cffi` library for high performance. It allows you to mimic the behavior of popular web browsers, effectively bypassing anti-bot measures and accessing websites that might otherwise block you. For commercial support, visit [impersonate.pro](https://impersonate.pro).

## Key Features

*   **Browser Impersonation:**  Mimic TLS/JA3 and HTTP/2 fingerprints of various browsers, including Chrome, Safari, Firefox, and others, to evade detection.
*   **Blazing Fast Performance:** Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`, as demonstrated by [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:**  Uses a `requests`-like API for easy adoption, reducing the learning curve.
*   **Pre-compiled Binaries:**  No need to compile on your machine, simplifying installation and setup.
*   **Asynchronous Support:**  Offers `asyncio` support with proxy rotation for efficient asynchronous web requests.
*   **HTTP/2 & HTTP/3 Support:** Supports the latest HTTP protocols, including HTTP/2 and HTTP/3, which some other libraries lack.
*   **WebSockets Integration:** Supports WebSocket connections for real-time data streaming.
*   **MIT License:**  Open-source and freely available for use.

### Feature Comparison

| Feature         | requests | aiohttp | httpx | pycurl | curl_cffi |
|-----------------|----------|---------|-------|--------|-----------|
| HTTP/2          | ❌       | ❌       | ✅    | ✅     | ✅        |
| HTTP/3          | ❌       | ❌       | ❌    | ☑️<sup>1</sup> | ✅<sup>2</sup>     |
| Synchronous     | ✅       | ❌       | ✅    | ✅     | ✅        |
| Asynchronous    | ❌       | ✅      | ✅    | ❌     | ✅        |
| WebSockets      | ❌       | ✅      | ❌    | ❌     | ✅        |
| Browser Fingerprints | ❌       | ❌      | ❌    | ❌     | ✅        |
| Speed           | 🐇       | 🐇🐇     | 🐇    | 🐇🐇   | 🐇🐇      |

*Notes:*
1.  For pycurl, you need an http/3 enabled libcurl.
2.  Since v0.11.4.

## Installation

Install `curl_cffi` with pip:

```bash
pip install curl_cffi --upgrade
```

This should work seamlessly on Linux, macOS, and Windows. If you encounter issues, you might need to compile and install `curl-impersonate` first and set appropriate environment variables, such as `LD_LIBRARY_PATH`.

To install beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

To install unstable version from GitHub:

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

On macOS, ensure you have the necessary dependencies:

```bash
brew install zstd nghttp2
```

## Usage

`curl_cffi` provides both a low-level `curl` API and a high-level, `requests`-like API.

### Requests-like API

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Use the latest browser version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")

# Randomly choose a browser version (Pro feature)
r = curl_cffi.get("https://example.com", impersonate="realworld")

# Pin a specific version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Custom fingerprints
r = curl_cffi.get("https://tls.browserleaks.com/json", ja3=..., akamai=...)

# Proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)

proxies = {"https": "socks://localhost:3128"}
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

`curl_cffi` supports a wide range of browser versions, as detailed in the [fork's documentation](https://github.com/lexiforest/curl-impersonate). For comprehensive browser fingerprint databases and advanced features, consider commercial support from [impersonate.pro](https://impersonate.pro).

| Browser        | Open Source                                                                        | Pro Version                                                                    |
|----------------|------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Chrome         | chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116<sup>[1]</sup>, chrome119<sup>[1]</sup>, chrome120<sup>[1]</sup>, chrome123<sup>[3]</sup>, chrome124<sup>[3]</sup>, chrome131<sup>[4]</sup>, chrome133a<sup>[5][6]</sup>, chrome136<sup>[6]</sup> | chrome132, chrome134, chrome135                                                  |
| Chrome Android | chrome99_android, chrome131_android <sup>[4]</sup>                                 | chrome132_android, chrome133_android, chrome134_android, chrome135_android           |
| Chrome iOS     | N/A                                                                                | coming soon                                                                      |
| Safari         | safari153 <sup>[2]</sup>, safari155 <sup>[2]</sup>, safari170 <sup>[1]</sup>, safari180 <sup>[4]</sup>, safari184 <sup>[6]</sup>, safari260 <sup>[8]</sup>                                                        | coming soon                                                                      |
| Safari iOS     | safari172_ios<sup>[1]</sup>, safari180_ios<sup>[4]</sup>, safari184_ios <sup>[6]</sup>, safari260_ios <sup>[8]</sup>                                                   | coming soon                                                                      |
| Firefox        | firefox133<sup>[5]</sup>, firefox135<sup>[7]</sup>                                 | coming soon                                                                      |
| Firefox Android| N/A                                                                                | firefox135_android                                                               |
| Tor            | tor145 <sup>[7]</sup>                                                                | coming soon                                                                      |
| Edge           | edge99, edge101                                                                    | edge133, edge135                                                                 |
| Opera          | N/A                                                                                | coming soon                                                                      |
| Brave          | N/A                                                                                | coming soon                                                                      |

*Notes:*
1.  Added in version `0.6.0`.
2.  Fixed in version `0.6.0`.
3.  Added in version `0.7.0`.
4.  Added in version `0.8.0`.
5.  Added in version `0.9.0`.
6.  The version postfix `-a`(e.g. `chrome133a`) means that this is an alternative version.
7.  Added in version `0.10.0`.
8.  Added in version `0.11.0`.

### Asyncio

```python
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

More Concurrency:

```python
import asyncio
from curl_cffi import AsyncSession

urls = [
    "https://google.com/",
    "https://facebook.com/",
    "https://twitter.com/",
]

async with AsyncSession() as s:
    tasks = []
    for url in urls:
        task = s.get(url)
        tasks.append(task)
    results = await asyncio.gather(*tasks)
```

Refer to the [docs](https://curl-cffi.readthedocs.io) for low-level APIs, Scrapy integration, and more.

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

*   **Scrapy Integration:**  `divtiply/scrapy-curl-cffi`, `jxlil/scrapy-impersonate`, and `tieyongjie/scrapy-fingerprint`.
*   **Adapters:**  `el1s7/curl-adapter` (for requests), `vgavro/httpx-curl-cffi` (for httpx).
*   **Captcha Resolvers:**  [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Acknowledgements

*   Inspired by `multippt/python_curl_cffi` (MIT license).
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py), (BSD license).
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).

## Contributing

When submitting a pull request, please create a branch other than `main` and check the "Allow edits by maintainers" box. Thank you!