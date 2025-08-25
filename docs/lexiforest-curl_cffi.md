# curl_cffi: The Fastest & Most Powerful Python HTTP Client with Browser Impersonation

[![PyPI Downloads](https://static.pepy.tech/badge/curl-cffi/week)](https://pepy.tech/projects/curl-cffi)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi)
[![Generic badge](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Generic badge](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io) | [Original Repo](https://github.com/lexiforest/curl_cffi)

`curl_cffi` is the ultimate Python library for high-performance HTTP requests, offering browser impersonation to bypass anti-bot defenses and access websites seamlessly.

## Key Features:

*   **Browser Impersonation:** Mimics TLS/JA3 and HTTP/2 fingerprints of popular browsers (Chrome, Safari, Firefox, Edge, and more) to bypass bot detection.
*   **Blazing Fast Performance:** Significantly faster than `requests` and `httpx`, on par with `aiohttp` and `pycurl` for rapid data retrieval.
*   **Familiar API:**  Uses a `requests`-like API for easy integration and minimal learning curve.
*   **Asynchronous Support:** Built-in `asyncio` support with proxy rotation for efficient concurrent requests.
*   **HTTP/2 & HTTP/3 Support:**  Offers full support for modern HTTP protocols, unavailable in `requests`.
*   **Websocket Support:** Includes synchronous and asynchronous websocket client.
*   **Pre-compiled:** No need to compile on your machine.
*   **Comprehensive Browser Fingerprints:** Supports numerous browser versions, including Chrome, Safari, Firefox, Edge, and more (see Supported Browsers below).

## Why Choose curl_cffi?

| Feature          | requests | aiohttp | httpx  | pycurl | curl_cffi |
|------------------|----------|---------|--------|--------|-----------|
| HTTP/2           | ❌       | ❌      | ✅     | ✅     | ✅         |
| HTTP/3           | ❌       | ❌      | ❌     | ✅     | ✅         |
| Sync             | ✅       | ❌      | ✅     | ✅     | ✅         |
| Async            | ❌       | ✅      | ✅     | ❌     | ✅         |
| Websocket        | ❌       | ✅      | ❌     | ❌     | ✅         |
| Fingerprints     | ❌       | ❌      | ❌     | ❌     | ✅         |
| Speed            | 🐇       | 🐇🐇    | 🐇     | 🐇🐇   | 🐇🐇        |

## Installation

```bash
pip install curl_cffi --upgrade
```

This installs pre-compiled binaries for Linux, macOS, and Windows.  If issues arise, you may need to install `curl-impersonate` and set environment variables.  For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

## Usage

### Basic Requests (requests-like API):

```python
import curl_cffi

r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Impersonate a specific browser version:
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Use proxies:
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions:

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
```

### Supported Browsers

`curl_cffi` supports browser versions compatible with the [curl-impersonate fork](https://github.com/lexiforest/curl-impersonate). See the [docs on impersonation](https://curl-cffi.readthedocs.io/en/latest/impersonate/_index.html) for details.

| Browser          | Open Source                                                     | Pro Version     |
|------------------|-----------------------------------------------------------------|-----------------|
| Chrome           | chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116, chrome119, chrome120, chrome123, chrome124, chrome131, chrome133a, chrome136 | chrome132, chrome134, chrome135     |
| Chrome Android   | chrome99_android, chrome131_android                              | chrome132_android, chrome133_android, chrome134_android, chrome135_android         |
| Safari           | safari153, safari155, safari170, safari180, safari184, safari260                       | coming soon     |
| Safari iOS       | safari172_ios, safari180_ios, safari184_ios, safari260_ios                        | coming soon     |
| Firefox          | firefox133, firefox135                        | coming soon     |
| Edge             | edge99, edge101                        | edge133, edge135     |
| Tor              | tor145                        | coming soon     |

### Asynchronous Usage (asyncio):

```python
from curl_cffi import AsyncSession
import asyncio

async def fetch_data(url):
    async with AsyncSession() as s:
        r = await s.get(url)
        return r.text()

async def main():
    results = await asyncio.gather(
        fetch_data("https://example.com"),
        fetch_data("https://google.com"),
    )
    print(results)

asyncio.run(main())
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

## Ecosystem & Integrations

*   **Scrapy Integration:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint].
*   **requests & httpx Adapters:**  [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi).
*   **Captcha Resolvers:**  [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Sponsorship and Support

This project is maintained by [contributors](https://github.com/lexiforest/curl_cffi/graphs/contributors) and [sponsors](https://github.com/sponsors/lexiforest).

## Contributing

Contribute to the project by creating a pull request on a new branch and check the "Allow edits by maintainers" box.