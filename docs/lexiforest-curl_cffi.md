# curl_cffi: Python Library for Advanced Web Scraping and Browser Impersonation

**Bypass bot detection and scrape the web with ease using `curl_cffi`, a powerful Python library for imitating browser fingerprints and accessing websites that would otherwise block you.**  [View the original repository](https://github.com/lexiforest/curl_cffi)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

`curl_cffi` is a Python binding for a [curl-impersonate fork](https://github.com/lexiforest/curl-impersonate) built with [cffi](https://cffi.readthedocs.io/en/latest/), allowing you to mimic browser behavior for web scraping, bypassing bot detection, and more.  For commercial support, visit [impersonate.pro](https://impersonate.pro).

## Key Features

*   **Browser Fingerprint Impersonation:**  Mimics TLS/JA3 and HTTP/2 fingerprints of various browsers (Chrome, Safari, Firefox, etc.) and even supports custom fingerprints.
*   **High Performance:** Significantly faster than `requests` and `httpx`, competitive with `aiohttp` and `pycurl`.  See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API for easy adoption.
*   **Pre-compiled:** No need to compile on your machine.
*   **Asynchronous Support:** Includes `asyncio` support with proxy rotation per request.
*   **HTTP/2 & HTTP/3 Support:** Supports both HTTP/2 and HTTP/3, unlike some other Python HTTP clients.
*   **WebSockets Support:** Supports WebSockets for real-time data retrieval.
*   **MIT Licensed:** Open-source and freely available for use.

## Why Choose curl_cffi?

| Feature        | requests | aiohttp | httpx | pycurl  | curl_cffi |
| -------------- | -------- | ------- | ----- | ------- | --------- |
| HTTP/2         | ❌       | ❌      | ✅    | ✅      | ✅        |
| HTTP/3         | ❌       | ❌      | ❌    | ☑️<sup>1</sup> | ✅<sup>2</sup>     |
| Synchronous    | ✅       | ❌      | ✅    | ✅      | ✅        |
| Asynchronous   | ❌       | ✅      | ✅    | ❌      | ✅        |
| WebSockets     | ❌       | ✅      | ❌    | ❌      | ✅        |
| Fingerprints   | ❌       | ❌      | ❌    | ❌      | ✅        |
| Speed          | 🐇       | 🐇🐇     | 🐇    | 🐇🐇     | 🐇🐇       |

Notes:
1.  For pycurl, you need an http/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2.  Since v0.11.4.

## Installation

Install `curl_cffi` using pip:

```bash
pip install curl_cffi --upgrade
```

This should work on Linux, macOS, and Windows out of the box. If you encounter issues, you might need to compile and install `curl-impersonate` first and set environment variables (e.g., `LD_LIBRARY_PATH`).

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

**macOS users:** You might need to install dependencies:

```bash
brew install zstd nghttp2
```

## Usage Examples

### requests-like API (v0.10)

```python
import curl_cffi

# Mimic Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Choose a specific Chrome version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Use proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)

proxies = {"https": "socks://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
r = s.get("https://httpbin.org/cookies")
print(r.json())
# {'cookies': {'foo': 'bar'}}
```

### Supported Browsers
`curl_cffi` supports a wide variety of browser versions. For more details on browser support and commercial offerings, visit [impersonate.pro](https://impersonate.pro)

[Browser Compatibility Table]


For more options, see the [docs on impersonation](https://curl-cffi.readthedocs.io/en/latest/impersonate.html).

### Asyncio

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

## Ecosystem Integrations

*   **Scrapy:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint)
*   **Adapters:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) (for `requests`), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) (for `httpx`)
*   **Captcha Solvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview)

## Sponsors

Maintenance of this project is made possible by all the [contributors](https://github.com/lexiforest/curl_cffi/graphs/contributors) and [sponsors](https://github.com/sponsors/lexiforest). If you'd like to sponsor this project and have your avatar or company logo appear below [click here](https://github.com/sponsors/lexiforest). 💖

------
<!-- SERP API -->
<a href="https://serpapi.com/" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/serpapi.png" alt="SerpAPI" height="67" width="63"></a>

Scrape Google and other search engines from [SerpApi](https://serpapi.com/)'s fast, easy, and complete API. 0.66s average response time (≤ 0.5s for Ludicrous Speed Max accounts), 99.95% SLAs, pay for successful responses only.
------

<!-- YES CAPTCHA -->
### Bypass Cloudflare with API

<a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>

Yescaptcha is a proxy service that bypasses Cloudflare and uses the API interface to
obtain verified cookies (e.g. `cf_clearance`). Click [here](https://yescaptcha.com/i/stfnIO)
to register: https://yescaptcha.com/i/stfnIO

------

<!-- CAPSOLVER -->
### Easy Captcha Bypass for Scraping

<a href="https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/capsolver.jpg" alt="CapSolver" height="50" width="178"></a>

[CapSolver](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC)
is an AI-powered tool that easily bypasses Captchas, allowing uninterrupted access to
public data. It supports a variety of Captchas and works seamlessly with `curl_cffi`,
Puppeteer, Playwright, and more. Fast, reliable, and cost-effective. Plus, `curl_cffi`
users can use the code **"CURL"** to get an extra 6% balance! and register
[here](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC).

------

## Acknowledgements

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi), which is under the MIT license.
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py), which is under the BSD license.
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).