# curl_cffi: Mimic Browser Fingerprints with Python & libcurl-impersonate

**Bypass website restrictions and scrape with ease using `curl_cffi`, the powerful Python library that impersonates browser fingerprints to access web content like a real user.** ([Original Repo](https://github.com/lexiforest/curl_cffi))

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` is a Python binding for the `curl-impersonate` fork, providing a fast and flexible way to make HTTP requests. It excels at bypassing bot detection and accessing websites that block standard Python HTTP clients.  Commercial support is available at [impersonate.pro](https://impersonate.pro).

## Key Features:

*   **Browser Impersonation:**  Mimics TLS/JA3 and HTTP/2 fingerprints of popular browsers (Chrome, Safari, Firefox, Edge, etc.) and supports custom fingerprints.
*   **High Performance:**  Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`. See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API for easy integration.
*   **Asynchronous Support:** Built-in `asyncio` support with proxy rotation.
*   **Modern Protocol Support:** Includes support for HTTP/2 and HTTP/3.
*   **Websocket Support:** Supports both synchronous and asynchronous WebSockets.
*   **Pre-compiled Binaries:**  No need to compile on your machine (usually).

##  Installation

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

For unstable GitHub versions:

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

On macOS, you might need:

```bash
brew install zstd nghttp2
```

##  Usage

`curl_cffi` offers both a low-level `curl` API and a high-level, `requests`-like API.

### Requests-like API (v0.10+)

```python
import curl_cffi

# Basic GET request impersonating Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Pin specific browser versions
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
```

See the original [README](https://github.com/lexiforest/curl_cffi) for more example.

### Supported Impersonation Profiles

`curl_cffi` supports a wide range of browser profiles.

| Browser           | Open Source                                                                                                        | Pro Version                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| Chrome            | chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116, chrome119, chrome120, chrome123, chrome124, chrome131, chrome133a, chrome136| chrome132, chrome134, chrome135                                         |
| Chrome Android    | chrome99_android, chrome131_android                                                                              | chrome132_android, chrome133_android, chrome134_android, chrome135_android |
| Chrome iOS        | N/A                                                                                                                | coming soon                                                               |
| Safari            | safari153, safari155, safari170, safari180, safari184, safari260                                                  | coming soon                                                               |
| Safari iOS        | safari172_ios, safari180_ios, safari184_ios, safari260_ios                                                       | coming soon                                                               |
| Firefox           | firefox133, firefox135                                                                                             | coming soon                                                               |
| Firefox Android   | N/A                                                                                                                | firefox135_android                                                        |
| Tor               | tor145                                                                                                             | coming soon                                                               |
| Edge              | edge99, edge101                                                                                                    | edge133, edge135                                                          |
| Opera             | N/A                                                                                                                | coming soon                                                               |
| Brave             | N/A                                                                                                                | coming soon                                                               |

For details, see the original [README](https://github.com/lexiforest/curl_cffi).

### Asyncio

```python
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

### Websockets

```python
from curl_cffi import WebSocket

def on_message(ws: WebSocket, message: str | bytes):
    print(message)

ws = WebSocket(on_message=on_message)
ws.run_forever("wss://api.gemini.com/v1/marketdata/BTCUSD")
```

## Ecosystem

*   **Scrapy Integration:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapters for Existing Libraries:**  [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) (requests adapter), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) (httpx adapter).
*   **Captcha Resolvers:**  [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Acknowledgements

*   Forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py).
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).