# curl_cffi: Python Binding for Advanced Web Scraping and Impersonation

**Bypass website restrictions and access content with ease using `curl_cffi`, the fastest and most versatile Python library for HTTP requests with browser impersonation.** [(Back to original repo)](https://github.com/lexiforest/curl_cffi)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)
[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` is a powerful Python library built on the [curl-impersonate fork](https://github.com/lexiforest/curl-impersonate), offering unparalleled control over HTTP requests and browser impersonation via [cffi](https://cffi.readthedocs.io/en/latest/).  It's the go-to solution for developers facing website restrictions or needing advanced features like browser fingerprinting. For commercial support, visit [impersonate.pro](https://impersonate.pro).

**Key Features:**

*   **Browser Impersonation:** Mimic various browsers (Chrome, Safari, Firefox, Edge, etc.) and their versions, including JA3/TLS and HTTP/2 fingerprints, to bypass anti-bot measures.
*   **High Performance:** Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl`. See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API for easy integration and a low learning curve.
*   **Pre-compiled and Ready to Use:** No need to compile anything on your machine; pre-compiled wheels are available.
*   **Asyncio Support:** Seamlessly integrates with `asyncio` for asynchronous requests and proxy rotation.
*   **HTTP/2 & HTTP/3 Support:**  Supports modern HTTP protocols, including HTTP/2 and HTTP/3.
*   **WebSocket Support:**  Includes both synchronous and asynchronous WebSocket functionalities.
*   **Customizable:** Supports custom JA3 and Akamai fingerprints for advanced users.
*   **MIT License:** Open-source and freely available for use.

## Installation

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

For unstable version from GitHub:

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

On macOS, you may need to install the following dependencies:

```bash
brew install zstd nghttp2
```

## Usage

### requests-like API (v0.10+)

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Impersonate a specific Chrome version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Randomly choose a browser version (Pro Feature)
r = curl_cffi.get("https://example.com", impersonate="realworld")

# Use custom fingerprints
r = curl_cffi.get("https://tls.browserleaks.com/json", ja3=..., akamai=...)

# Use proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()

# Set cookies
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)

# Retrieve cookies
r = s.get("https://httpbin.org/cookies")
print(r.json())
```

### Supported Browsers

`curl_cffi` supports a wide range of browser versions.

| Browser          | Open Source                                         | Pro Version                        |
| ---------------- | --------------------------------------------------- | ---------------------------------- |
| Chrome           | chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116<sup>[1]</sup>, chrome119<sup>[1]</sup>, chrome120<sup>[1]</sup>, chrome123<sup>[3]</sup>, chrome124<sup>[3]</sup>, chrome131<sup>[4]</sup>, chrome133a<sup>[5][6]</sup>, chrome136<sup>[6]</sup>          | chrome132, chrome134, chrome135  |
| Chrome Android   | chrome99_android, chrome131_android <sup>[4]</sup>         | chrome132_android, chrome133_android, chrome134_android, chrome135_android |
| Safari           | safari153 <sup>[2]</sup>, safari155 <sup>[2]</sup>, safari170 <sup>[1]</sup>, safari180 <sup>[4]</sup>, safari184 <sup>[6]</sup>, safari260 <sup>[8]</sup>| coming soon |
| Safari iOS       | safari172_ios<sup>[1]</sup>, safari180_ios<sup>[4]</sup>, safari184_ios <sup>[6]</sup>, safari260_ios <sup>[8]</sup>| coming soon |
| Firefox          | firefox133<sup>[5]</sup>, firefox135<sup>[7]</sup>                                        | coming soon                        |
| Firefox Android  | N/A                                               | firefox135_android               |
| Tor              | tor145 <sup>[7]</sup>                                  | coming soon                        |
| Edge             | edge99, edge101                                      | edge133, edge135                   |

For detailed browser version support, please refer to the [original repo](https://github.com/lexiforest/curl_cffi).

### Asyncio

```python
from curl_cffi import AsyncSession
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com")
    # ...
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

`curl_cffi` integrates seamlessly with other popular tools:

*   **Scrapy:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapters:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi).
*   **Captcha Solvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview). (See the original README for promotional codes.)

## Acknowledgements

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py).
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).