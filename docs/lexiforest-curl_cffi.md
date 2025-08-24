# curl_cffi: The Ultimate Python Library for Web Scraping and Browser Impersonation

[![PyPI Downloads](https://static.pepy.tech/badge/curl-cffi/week)](https://pepy.tech/projects/curl-cffi)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://img.shields.io/pypi/pyversions/curl_cffi)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

**Bypass website restrictions and scrape the web like a pro with `curl_cffi`, a powerful Python binding for `curl` that lets you impersonate browsers and handle advanced HTTP features!** Access the official [documentation](https://curl-cffi.readthedocs.io) for comprehensive guides and tutorials.

`curl_cffi` is a versatile Python library built on top of the [curl-impersonate fork](https://github.com/lexiforest/curl-impersonate) and uses [cffi](https://cffi.readthedocs.io/en/latest/). For commercial support, visit [impersonate.pro](https://impersonate.pro).

## Key Features:

*   **Browser Impersonation:** Mimic various browsers' TLS/JA3 and HTTP/2 fingerprints, including Chrome, Safari, Firefox, and more.  Effectively bypass anti-bot measures.
*   **Blazing Fast Performance:**  Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`.  See the [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API for easy adoption.
*   **Pre-compiled:** No need to compile on your machine, simplifying installation.
*   **Asynchronous Support:**  Built-in `asyncio` support with proxy rotation for asynchronous operations.
*   **HTTP/2 & HTTP/3 Support:** Supports modern HTTP protocols, unlike `requests`.
*   **WebSockets:**  Includes support for WebSockets.
*   **Comprehensive Fingerprint Database:**  Provides a wide range of browser fingerprints and custom fingerprint capabilities.
*   **MIT License:**  Free to use and distribute.

| Feature | requests | aiohttp | httpx | pycurl | curl_cffi |
|---|---|---|---|---|---|
| HTTP/2 | ❌ | ❌ | ✅ | ✅ | ✅ |
| HTTP/3 | ❌ | ❌ | ❌ | ☑️<sup>1</sup> | ✅<sup>2</sup> |
| Sync | ✅ | ❌ | ✅ | ✅ | ✅ |
| Async | ❌ | ✅ | ✅ | ❌ | ✅ |
| WebSocket | ❌ | ✅ | ❌ | ❌ | ✅ |
| Fingerprints | ❌ | ❌ | ❌ | ❌ | ✅ |
| Speed | 🐇 | 🐇🐇 | 🐇 | 🐇🐇 | 🐇🐇 |

**Notes:**

1.  Requires an HTTP/3-enabled `libcurl`.
2.  Available since v0.11.4.

## Installation

Install `curl_cffi` using pip:

```bash
pip install curl_cffi --upgrade
```

This should work on most platforms.  If you encounter issues, you may need to compile `curl-impersonate` separately and set environment variables like `LD_LIBRARY_PATH`.

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

Or, to install the latest development version from GitHub:

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

On macOS, you may need to install dependencies:

```bash
brew install zstd nghttp2
```

## Usage

`curl_cffi` offers both low-level `curl` and a high-level `requests`-like API.

### Requests-like API

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Impersonate a specific Chrome version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Impersonate based on real-world market share (Pro Feature)
r = curl_cffi.get("https://example.com", impersonate="realworld")

# Use custom JA3/akamai fingerprints
r = curl_cffi.get("https://tls.browserleaks.com/json", ja3=..., akamai=...)

# Use Proxies
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

### Supported Browser Fingerprints

`curl_cffi` supports a wide range of browser fingerprints. See the table below. Commercial support provides more browser fingerprints via [impersonate.pro](https://impersonate.pro). Further details on impersonation are found in the [docs on impersonation](https://curl-cffi.readthedocs.io/en/latest/impersonate/_index.html).

| Browser          | Open Source                                                                                                 | Pro Version                                                                                                                                                              |
| :--------------- | :---------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chrome           | chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116<sup>[1]</sup>, chrome119<sup>[1]</sup>, chrome120<sup>[1]</sup>, chrome123<sup>[3]</sup>, chrome124<sup>[3]</sup>, chrome131<sup>[4]</sup>, chrome133a<sup>[5][6]</sup>, chrome136<sup>[6]</sup> | chrome132, chrome134, chrome135                                                                                                                                              |
| Chrome Android   | chrome99_android, chrome131_android <sup>[4]</sup>                                                         | chrome132_android, chrome133_android, chrome134_android, chrome135_android                                                                                               |
| Chrome iOS       | N/A                                                                                                         | coming soon                                                                                                                                                             |
| Safari <sup>[7]</sup>   | safari153 <sup>[2]</sup>, safari155 <sup>[2]</sup>, safari170 <sup>[1]</sup>, safari180 <sup>[4]</sup>, safari184 <sup>[6]</sup>, safari260 <sup>[8]</sup>                                                                                                  | coming soon                                                                                                                                                             |
| Safari iOS <sup>[7]</sup> | safari172_ios<sup>[1]</sup>, safari180_ios<sup>[4]</sup>, safari184_ios <sup>[6]</sup>, safari260_ios <sup>[8]</sup>                                                                                                  | coming soon                                                                                                                                                             |
| Firefox          | firefox133<sup>[5]</sup>, firefox135<sup>[7]</sup>                                                        | coming soon                                                                                                                                                             |
| Firefox Android  | N/A                                                                                                         | firefox135_android                                                                                                                                                        |
| Tor              | tor145 <sup>[7]</sup>                                                                                       | coming soon                                                                                                                                                             |
| Edge             | edge99, edge101                                                                                             | edge133, edge135                                                                                                                                                          |
| Opera            | N/A                                                                                                         | coming soon                                                                                                                                                             |
| Brave            | N/A                                                                                                         | coming soon                                                                                                                                                             |

**Notes:**

1.  Added in version `0.6.0`.
2.  Fixed in version `0.6.0`.
3.  Added in version `0.7.0`.
4.  Added in version `0.8.0`.
5.  Added in version `0.9.0`.
6.  The version postfix `-a` denotes an alternative version.
7.  Added in version `0.10.0`.
8.  Added in version `0.11.0`.
9.  Since `0.11.0`, the format `safari184_ios` is preferred over `safari18_4_ios`.
10. Added in `0.12.0`.

### Asyncio

```python
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

More concurrency:

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

For low-level APIs, Scrapy integration and other advanced topics, see the [docs](https://curl-cffi.readthedocs.io).

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

*   **Scrapy Integration:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapter Integrations:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) (requests), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) (httpx).
*   **Captcha Resolvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Acknowledgements

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi) (MIT License).
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py) (BSD License).
*   Asyncio support inspired by Tornado's curl HTTP client.
*   Synchronous WebSocket API inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   Asynchronous WebSocket API inspired by [aiohttp](https://github.com/aio-libs/aiohttp).

## Contributing

Contributions are welcome! Please use a separate branch and check the "Allow edits by maintainers" box when submitting a pull request to help with linting and style fixes.  Thank you!

[Back to Top](#curl_cffi) - Visit the original repository on [GitHub](https://github.com/lexiforest/curl_cffi) for more details.