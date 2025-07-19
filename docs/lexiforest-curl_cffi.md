# curl_cffi: Python Binding for Powerful Web Scraping & Browser Impersonation

**Bypass website restrictions and scrape the web like a pro with `curl_cffi`, the most popular Python binding for `curl`.**  ([Original Repo](https://github.com/lexiforest/curl_cffi))

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Generic badge](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Generic badge](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io) | [Commercial Support: impersonate.pro](https://impersonate.pro)

## Key Features

*   **Browser Impersonation:** Emulates browser fingerprints (TLS/JA3, HTTP/2) to bypass anti-scraping measures.
*   **High Performance:** Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`.
*   **Familiar API:**  Uses a `requests`-like API for easy adoption.
*   **Asynchronous Support:** Offers `asyncio` integration with proxy rotation.
*   **HTTP/2 & HTTP/3 Support:** Compatible with modern web protocols.
*   **WebSockets:**  Includes WebSocket support for real-time data streams.
*   **Pre-compiled:**  No need to compile on your machine.

## Why Choose curl_cffi?

Are you facing blocks from websites? `curl_cffi` can help you by impersonating browsers, allowing you to scrape data that would otherwise be inaccessible.

## Installation

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

For unstable versions:

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

## Usage

`curl_cffi` offers both a low-level `curl` API and a high-level, `requests`-like API.

### Requests-like API (v0.10+)

```python
import curl_cffi

# Impersonate a browser (e.g., Chrome)
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Use specific browser versions
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Randomly choose a browser version (Pro feature)
r = curl_cffi.get("https://example.com", impersonate="realworld")

# Use proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()

# Get cookies from a server
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)

# Retrieve the cookies
r = s.get("https://httpbin.org/cookies")
print(r.json())
```

### Supported Impersonation Browsers

`curl_cffi` supports a wide range of browser versions.  For commercial support with a comprehensive database, visit [impersonate.pro](https://impersonate.pro). See the original README for the full list of supported browsers.

### Asyncio

```python
from curl_cffi import AsyncSession
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com")
    print(r.text)
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

*   **Scrapy Integration:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint)
*   **Adapters:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) (requests), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) (httpx)
*   **Captcha Resolvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview)

## Sponsors & Acknowledgements

Thank you to the [contributors](https://github.com/lexiforest/curl_cffi/graphs/contributors) and [sponsors](https://github.com/sponsors/lexiforest) who make this project possible!

This project is originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi) and inspired by many other open-source projects.