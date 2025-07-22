# curl_cffi: The Fastest Python HTTP Client with Browser Impersonation

**Bypass website restrictions and scrape data like a pro with `curl_cffi`, a powerful Python library that mimics browser fingerprints for advanced web scraping and testing.  [Check out the original repo!](https://github.com/lexiforest/curl_cffi)**

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)
[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` is the premier Python binding for the `curl-impersonate` fork, enabling you to seamlessly mimic browser behavior. It's a faster and more versatile alternative to `requests` and `httpx`, designed to overcome website anti-scraping measures. Commercial support is available at [impersonate.pro](https://impersonate.pro).

**Key Features:**

*   **Browser Impersonation:**  Mimics TLS/JA3 and HTTP/2 fingerprints of popular browsers, including Chrome, Safari, Firefox, and more, to bypass anti-bot defenses.
*   **Blazing Fast Performance:**  Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl`.
*   **Familiar API:**  Offers a user-friendly API inspired by the `requests` library, minimizing the learning curve.
*   **Asynchronous Support:** Includes full `asyncio` support with proxy rotation for efficient, concurrent requests.
*   **HTTP/2 & HTTP/3 and WebSocket Support:**  Supports modern protocols not available in all Python HTTP clients.
*   **Pre-compiled and Ready to Use:**  Provides pre-compiled binaries for easy installation across various platforms.

------

**Bypass Cloudflare with API**

<a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>

Yescaptcha is a proxy service that bypasses Cloudflare and uses the API interface to obtain verified cookies (e.g. `cf_clearance`). Click [here](https://yescaptcha.com/i/stfnIO) to register.

------

**Easy Captcha Bypass for Scraping**

<a href="https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/capsolver.jpg" alt="CapSolver" height="50" width="178"></a>

[CapSolver](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC)
is an AI-powered tool that easily bypasses Captchas, allowing uninterrupted access to
public data. It supports a variety of Captchas and works seamlessly with `curl_cffi`,
Puppeteer, Playwright, and more. Fast, reliable, and cost-effective. Plus, `curl_cffi`
users can use the code **"CURL"** to get an extra 6% balance! and register
[here](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC).

------

## Install

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

### requests-like API (v0.10+)

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Pin a specific version (e.g., Chrome 124)
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# With proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
```

### Asyncio

```python
from curl_cffi import AsyncSession

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

## Ecosystem

*   [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi)
*   [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate)
*   [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint)
*   [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter)
*   [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi)
*   [CapSolver](https://docs.capsolver.com/en/api/)
*   [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview)

## Supported Browsers

The open source version of `curl_cffi` supports a variety of browser versions.

*   **Chrome:**  `chrome99`, `chrome100`, `chrome101`, `chrome104`, `chrome107`, `chrome110`, `chrome116`, `chrome119`, `chrome120`, `chrome123`, `chrome124`, `chrome131`, `chrome133a`, `chrome136`
*   **Chrome Android:** `chrome99_android`, `chrome131_android`
*   **Safari:** `safari153`, `safari155`, `safari170`, `safari180`, `safari184`, `safari260`
*   **Safari iOS:** `safari172_ios`, `safari180_ios`, `safari184_ios`, `safari260_ios`
*   **Firefox:** `firefox133`, `firefox135`
*   **Tor:** `tor145`
*   **Edge:** `edge99`, `edge101`

Commercial Support and Additional Browser Versions are available from [impersonate.pro](https://impersonate.pro).

## Acknowledgement

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py).
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).