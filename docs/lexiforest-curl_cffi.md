# curl_cffi: The Fastest Python HTTP Client with Browser Impersonation

**Bypass website restrictions and access the web with ease using `curl_cffi`, the Python library that lets you impersonate browsers and offers unparalleled speed.**

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi.svg)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io) | [Original Repo](https://github.com/lexiforest/curl_cffi)

Built upon a [curl-impersonate fork](https://github.com/lexiforest/curl-impersonate) and [cffi](https://cffi.readthedocs.io/en/latest/), `curl_cffi` is a powerful Python library that mimics browser fingerprints for seamless web interactions. Ideal for bypassing anti-bot measures, it provides a high-performance alternative to libraries like `requests` and `httpx`. For commercial support, visit [impersonate.pro](https://impersonate.pro).

## Key Features

*   **Browser Impersonation:** Easily mimic the TLS/JA3 and HTTP/2 fingerprints of popular browsers (Chrome, Safari, Firefox, etc.) and customize them.
*   **High Performance:** Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`. See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Mimics the `requests` API for easy integration and a quick learning curve.
*   **Pre-compiled:** No need to compile on your machine.
*   **Asynchronous Support:** Supports `asyncio` with proxy rotation on each request.
*   **Modern Protocol Support:** Includes HTTP/2 and HTTP/3 support, features missing in `requests`.
*   **WebSocket Support:** Both synchronous and asynchronous WebSocket support.
*   **MIT License:** Free to use and integrate in your projects.

## Performance Comparison

| Feature        | requests | aiohttp | httpx  | pycurl | curl_cffi |
| -------------- | -------- | ------- | ------ | ------ | --------- |
| HTTP/2         | ❌       | ❌      | ✅     | ✅     | ✅        |
| HTTP/3         | ❌       | ❌      | ❌     | ☑️<sup>1</sup> | ✅<sup>2</sup>       |
| Sync           | ✅       | ❌      | ✅     | ✅     | ✅        |
| Async          | ❌       | ✅      | ✅     | ❌     | ✅        |
| WebSocket      | ❌       | ✅      | ❌     | ❌     | ✅        |
| Fingerprints   | ❌       | ❌      | ❌     | ❌     | ✅        |
| Speed          | 🐇       | 🐇🐇     | 🐇     | 🐇🐇     | 🐇🐇      |

Notes:

1.  Requires HTTP/3-enabled libcurl.
2.  Supported since v0.11.4.

## Installation

```bash
pip install curl_cffi --upgrade
```

This typically works out-of-the-box on Linux, macOS, and Windows.
If you encounter issues, you may need to compile and install `curl-impersonate` separately and configure environment variables like `LD_LIBRARY_PATH`.

To install beta releases:

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

macOS users may need to install these dependencies:

```bash
brew install zstd nghttp2
```

## Usage

`curl_cffi` offers both a low-level `curl` API and a high-level, `requests`-like API.

### requests-like

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Use latest browser version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")

# Pin specific versions
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Customize fingerprints
r = curl_cffi.get("https://tls.browserleaks.com/json", ja3=..., akamai=...)

# Use proxies
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

### Supported Browser Versions

`curl_cffi` supports the browser versions compatible with the [curl-impersonate](https://github.com/lwthiker/curl-impersonate) fork.

Open source version of curl_cffi includes versions whose fingerprints differ from previous versions.
If you see a version, e.g. `chrome135`, were skipped, you can simply impersonate it with your own headers and the previous version.

For comprehensive browser fingerprints, consider [impersonate.pro](https://impersonate.pro).

| Browser         | Open Source                                                                                                                                                                | Pro version  |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| Chrome          | chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116<sup>[1]</sup>, chrome119<sup>[1]</sup>, chrome120<sup>[1]</sup>, chrome123<sup>[3]</sup>, chrome124<sup>[3]</sup>, chrome131<sup>[4]</sup>, chrome133a<sup>[5][6]</sup>, chrome136<sup>[6]</sup> | chrome132, chrome134, chrome135 |
| Chrome Android  | chrome99_android, chrome131_android <sup>[4]</sup>                                                                                                                           | chrome132_android, chrome133_android, chrome134_android, chrome135_android |
| Chrome iOS      | N/A                                                                                                                                                                        | coming soon  |
| Safari <sup>[7]</sup>       | safari153 <sup>[2]</sup>, safari155 <sup>[2]</sup>, safari170 <sup>[1]</sup>, safari180 <sup>[4]</sup>, safari184 <sup>[6]</sup>, safari260 <sup>[8]</sup>          | coming soon  |
| Safari iOS <sup>[7]</sup>    | safari172_ios<sup>[1]</sup>, safari180_ios<sup>[4]</sup>, safari184_ios <sup>[6]</sup>, safari260_ios <sup>[8]</sup>                                                 | coming soon  |
| Firefox         | firefox133<sup>[5]</sup>, firefox135<sup>[7]</sup>                                                                                                                            | coming soon  |
| Firefox Android | N/A                                                                                                                                                                        | firefox135_android |
| Tor             | tor145 <sup>[7]</sup>                                                                                                                                                               | coming soon  |
| Edge            | edge99, edge101                                                                                                                                                              | edge133, edge135 |
| Opera           | N/A                                                                                                                                                                        | coming soon  |
| Brave           | N/A                                                                                                                                                                        | coming soon  |

Notes:
1. Added in version `0.6.0`.
2. Fixed in version `0.6.0`, previous http2 fingerprints were [not correct](https://github.com/lwthiker/curl-impersonate/issues/215).
3. Added in version `0.7.0`.
4. Added in version `0.8.0`.
5. Added in version `0.9.0`.
6. The version postfix `-a`(e.g. `chrome133a`) means that this is an alternative version, i.e. the fingerprint has not been officially updated by browser, but has been observed because of A/B testing.
5. Added in version `0.10.0`.
6. Added in version `0.11.0`.
7. Since `0.11.0`, the format `safari184_ios` is preferred over `safari18_4_ios`, both are supported, but the latter is quite confusing and hard to parse.
8. Added in  `0.12.0`.

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

For low-level APIs, Scrapy integration and other advanced topics, see the
[docs](https://curl-cffi.readthedocs.io) for more details.

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

*   **Scrapy Integration:**  [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapters for Existing Libraries:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) (requests), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) (httpx).
*   **Captcha Resolvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Acknowledgements

*   Based on [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi) (MIT license).
*   Headers/Cookies files from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py) (BSD license).
*   Asyncio support inspired by Tornado.
*   Synchronous WebSocket API inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   Asynchronous WebSocket API inspired by [aiohttp](https://github.com/aio-libs/aiohttp).

## Contributing

Please use a separate branch and check the "Allow edits by maintainers" box when submitting pull requests.