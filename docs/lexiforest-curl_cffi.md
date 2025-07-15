# curl_cffi: The Ultimate Python Library for Web Scraping and Browser Impersonation

**Bypass website restrictions and scrape the web like a pro with `curl_cffi`, the Python binding for `curl-impersonate`.**  [Visit the original repository](https://github.com/lexiforest/curl_cffi)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi)

*   [Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` offers a powerful and flexible way to interact with websites, including the ability to impersonate browsers for enhanced scraping capabilities. Built upon the `curl-impersonate` fork and using [cffi](https://cffi.readthedocs.io/en/latest/), `curl_cffi` provides a robust solution for web scraping, bypassing Cloudflare, and accessing websites that employ anti-bot measures.

## Key Features

*   **Browser Impersonation:** Mimics various browser fingerprints (TLS/JA3, HTTP/2) to bypass bot detection, including Chrome, Safari, Firefox, and more.
*   **High Performance:**  Significantly faster than popular Python HTTP clients like `requests` and `httpx`, comparable to `aiohttp` and `pycurl`.
*   **User-Friendly API:** Uses a familiar `requests`-like API for easy integration and a quick learning curve.
*   **Asynchronous Support:** Seamlessly integrates with `asyncio` for efficient concurrent requests and proxy rotation.
*   **HTTP/2 & HTTP/3 Support:**  Supports modern protocols that `requests` does not.
*   **WebSocket Support:** Includes synchronous and asynchronous WebSocket capabilities.
*   **Pre-compiled:** No need to compile on your machine.
*   **Comprehensive Browser Fingerprint Database**:  Supports a range of browsers and versions via the commercial support option.

## Bypass Cloudflare and Solve Captchas

*   **SerpApi:** Scrape Google and other search engines with ease.
    <a href="https://serpapi.com/" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/serpapi.png" alt="SerpAPI" height="67" width="63"></a>
*   **Yescaptcha:** Bypass Cloudflare with an API interface to obtain verified cookies (e.g. `cf_clearance`).
    <a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>
    [Register at Yescaptcha](https://yescaptcha.com/i/stfnIO)
*   **CapSolver:** An AI-powered captcha solver for uninterrupted access to public data.
    <a href="https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/capsolver.jpg" alt="CapSolver" height="50" width="178"></a>
    [Register at CapSolver](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC) and use code **"CURL"** for a 6% balance bonus!

## Installation

Install `curl_cffi` using pip:

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

To install from GitHub (unstable):

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

## Usage

### Requests-like API (v0.10+)

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Pin specific version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Use proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions (v0.10+)

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
r = s.get("https://httpbin.org/cookies")
print(r.json())
```

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

## Supported Browsers

`curl_cffi` supports a variety of browsers and versions. See the detailed list in the original README for all the specific browser version supported.

## Ecosystem Integration

*   [scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi)
*   [httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi)
*   Captcha resolvers (CapSolver and Yescaptcha)

## Acknowledgements

*   Forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
*   Headers/Cookies from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py).
*   Asyncio support inspired by Tornado.
*   WebSocket API inspired by websocket_client and aiohttp.