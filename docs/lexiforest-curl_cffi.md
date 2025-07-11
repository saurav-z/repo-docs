# curl_cffi: The Ultimate Python Library for Web Scraping and Browser Impersonation

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

**Bypass website restrictions and scrape the web with ease using `curl_cffi`, a powerful Python library that mimics browser fingerprints to avoid detection.**

**[View the original repository on GitHub](https://github.com/lexiforest/curl_cffi)**

## Key Features:

*   **Browser Impersonation:** Mimic various browsers' TLS/JA3 and HTTP/2 fingerprints (Chrome, Safari, Firefox, and more) to evade anti-scraping measures.
*   **High Performance:** Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`, see [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API, making it easy to learn and integrate.
*   **Asynchronous Support:** Includes `asyncio` support with proxy rotation for efficient, concurrent requests.
*   **HTTP/2 and HTTP/3 Support:** Supports modern HTTP protocols that `requests` doesn't.
*   **WebSockets:** Provides support for WebSocket connections.
*   **Pre-compiled:** No need for manual compilation on your machine, making installation simple.

## Installation

```bash
pip install curl_cffi --upgrade
```

## Usage Examples

### Requests-like API (v0.10+)

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Pin a specific version (e.g. Chrome 124)
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Use proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions
```python
s = curl_cffi.Session()

# httpbin is a http test website, this endpoint makes the server set cookies
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)

# retrieve cookies again to verify
r = s.get("https://httpbin.org/cookies")
print(r.json())
# {'cookies': {'foo': 'bar'}}
```

### Asyncio Example
```python
import asyncio
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

## Supported Browsers

`curl_cffi` supports browser versions compatible with the [curl-impersonate](https://github.com/lexiforest/curl-impersonate) fork.

For comprehensive browser fingerprints and commercial support, visit [impersonate.pro](https://impersonate.pro).

## Ecosystem

*   **Scrapy Integration:**  [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapters:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) (for `requests`), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) (for `httpx`).
*   **Captcha Resolvers:**  Integrates with [CapSolver](https://docs.capsolver.com/en/api/) and [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Sponsors

Maintenance of this project is made possible by all the [contributors](https://github.com/lexiforest/curl_cffi/graphs/contributors) and [sponsors](https://github.com/sponsors/lexiforest). If you'd like to sponsor this project and have your avatar or company logo appear, [click here](https://github.com/sponsors/lexiforest).

## External Services

### Scrape Google and other search engines
*   Scrape Google and other search engines from [SerpApi](https://serpapi.com/)'s fast, easy, and complete API. 0.66s average response time (≤ 0.5s for Ludicrous Speed Max accounts), 99.95% SLAs, pay for successful responses only.
    [![SerpApi](https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/serpapi.png)](https://serpapi.com/)

### Bypass Cloudflare with API
*   Yescaptcha is a proxy service that bypasses Cloudflare and uses the API interface to
    obtain verified cookies (e.g. `cf_clearance`). Click [here](https://yescaptcha.com/i/stfnIO)
    to register: https://yescaptcha.com/i/stfnIO
    [![Yes Captcha!](https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png)](https://yescaptcha.com/i/stfnIO)

### Easy Captcha Bypass for Scraping
*   [CapSolver](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC)
    is an AI-powered tool that easily bypasses Captchas, allowing uninterrupted access to
    public data. It supports a variety of Captchas and works seamlessly with `curl_cffi`,
    Puppeteer, Playwright, and more. Fast, reliable, and cost-effective. Plus, `curl_cffi`
    users can use the code **"CURL"** to get an extra 6% balance! and register
    [here](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC).
    [![CapSolver](https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/capsolver.jpg)](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC)

## Acknowledgements

*   Based on [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi)
*   Headers/Cookies files from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py)
*   Asyncio support inspired by Tornado's curl http client.
*   Synchronous WebSocket API inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   Asynchronous WebSocket API inspired by [aiohttp](https://github.com/aio-libs/aiohttp).