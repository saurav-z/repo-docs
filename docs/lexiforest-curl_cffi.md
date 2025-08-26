# curl_cffi: The Fastest Python HTTP Client for Impersonating Browsers

**`curl_cffi` provides a high-performance, user-friendly Python binding for the powerful `curl` library, enabling you to seamlessly impersonate browsers and bypass website restrictions.**  [See the original repo](https://github.com/lexiforest/curl_cffi).

[![PyPI Downloads](https://static.pepy.tech/badge/curl-cffi/week)](https://pepy.tech/projects/curl-cffi)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://img.shields.io/pypi/pyversions/curl_cffi)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io) | [Commercial Support](https://impersonate.pro)

## Key Features

*   **Browser Impersonation:** Mimics TLS/JA3 and HTTP/2 fingerprints of various browsers, including Chrome, Safari, and Firefox, to bypass bot detection.
*   **High Performance:** Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl`.  See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:**  Uses a `requests`-like API, making it easy to learn and use.
*   **Pre-compiled:** No need to compile dependencies on your machine; wheels are provided.
*   **Asynchronous Support:**  Seamless `asyncio` integration with proxy rotation capabilities.
*   **HTTP/2 & HTTP/3 Support:**  Full support for modern HTTP protocols, offering advantages over `requests`.
*   **Websocket Support:**  Includes both synchronous and asynchronous websocket APIs.
*   **Open Source and MIT Licensed:**  Available under the permissive MIT license.

## Why Choose curl_cffi?

Are you facing bot detection or IP blocks when scraping websites?  `curl_cffi` allows you to impersonate browsers to bypass Cloudflare and other anti-bot solutions.

## Bypassing Cloudflare

Integrate with captcha solving services to easily bypass Cloudflare.

**Yescaptcha:**  Obtain verified cookies using the API interface.
*   Register here:  https://yescaptcha.com/i/stfnIO
*   <a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>

## Installation

```bash
pip install curl_cffi --upgrade
```

For beta releases:

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

**Dependencies for macOS:**

```bash
brew install zstd nghttp2
```

## Usage

`curl_cffi` offers both a low-level `curl` API and a high-level, `requests`-compatible API.

###  `requests`-like API

```python
import curl_cffi

r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")  # latest Chrome

r = curl_cffi.get("https://example.com", impersonate="realworld")  # Random version

r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")  # specific version

# Customize fingerprints:
r = curl_cffi.get("https://tls.browserleaks.com/json", ja3=..., akamai=...)

# Proxy Support:
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

`curl_cffi` supports browser versions as defined by `curl-impersonate`.  See [impersonate.pro](https://impersonate.pro) for commercial support with an extensive browser fingerprint database.

| Browser          | Open Source                                                                                                                                                                                                          |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chrome           | chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116<sup>[1]</sup>, chrome119<sup>[1]</sup>, chrome120<sup>[1]</sup>, chrome123<sup>[3]</sup>, chrome124<sup>[3]</sup>, chrome131<sup>[4]</sup>, chrome133a<sup>[5][6]</sup>, chrome136<sup>[6]</sup>                                                                                                                                                                            |
| Chrome Android   | chrome99_android, chrome131_android <sup>[4]</sup>                                                                                                                                                                  |
| Safari           | safari153 <sup>[2]</sup>, safari155 <sup>[2]</sup>, safari170 <sup>[1]</sup>, safari180 <sup>[4]</sup>, safari184 <sup>[6]</sup>, safari260 <sup>[8]</sup>                                                                                                                                                                       |
| Safari iOS       | safari172_ios<sup>[1]</sup>, safari180_ios<sup>[4]</sup>, safari184_ios <sup>[6]</sup>, safari260_ios <sup>[8]</sup>                                                                                                                                                                         |
| Firefox          | firefox133<sup>[5]</sup>, firefox135<sup>[7]</sup>                                                                                                                                                                       |
| Edge             | edge99, edge101                                                                                                                                                                                                      |
| Tor              | tor145 <sup>[7]</sup>                                                                                                                                                                                                      |

*See the original README for more detailed notes on browser version support.*

###  Asynchronous Usage (`asyncio`)

```python
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

###  WebSockets

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

`curl_cffi` seamlessly integrates with several popular tools and services.

*   **Scrapy:**  [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapters:**  [el1s7/curl-adapter] (https://github.com/el1s7/curl-adapter) for requests, and [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) for httpx.
*   **Captcha Solvers:**  [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Acknowledgements

*   Forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
*   Headers/Cookies files from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py).
*   Asyncio support inspired by Tornado's curl http client.
*   WebSocket APIs are inspired by [websocket_client](https://github.com/websocket-client/websocket-client) and [aiohttp](https://github.com/aio-libs/aiohttp).

## Contributing

Please use a separate branch and enable "Allow edits by maintainers" when submitting pull requests.