# curl_cffi: Python's Ultimate Library for Web Scraping and Browser Impersonation

**Bypass web restrictions with ease using `curl_cffi`, the fastest and most versatile Python binding for `curl`, designed for advanced web scraping and browser impersonation. [Check out the original repo](https://github.com/lexiforest/curl_cffi)**

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=discord)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io) | [Commercial Support](https://impersonate.pro)

`curl_cffi` is the go-to Python library for developers needing to scrape websites, bypass anti-bot measures, or simply control HTTP requests with unparalleled flexibility and speed. It leverages the power of `curl-impersonate`, allowing you to mimic browser behavior and access content that would otherwise be blocked.

**Key Features:**

*   🚀 **Blazing Fast:** Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl` in performance.  See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   🎭 **Browser Impersonation:** Mimics browser fingerprints (TLS/JA3, HTTP/2) for effective web scraping and access to restricted content.  Supports a wide range of browsers and versions.
*   💻 **Familiar API:**  Offers a `requests`-like API for ease of use, minimizing the learning curve.
*   📦 **Pre-compiled:** No need to compile on your machine; ready to use out of the box.
*   🔄 **Asyncio Support:** Fully supports asynchronous operations, including proxy rotation for enhanced scraping.
*   🌐 **HTTP/2 & HTTP/3:** Supports the latest HTTP protocols, including HTTP/2 and HTTP/3, for modern web interactions.
*   📡 **WebSockets:** Includes robust WebSocket support for real-time communication.
*   🔑 **MIT License:** Free to use, modify, and distribute.

### Bypass Cloudflare and More

Leverage the following services for advanced capabilities:

#### Scrape Google and other search engines
Scrape Google and other search engines from [SerpApi](https://serpapi.com/)'s fast, easy, and complete API. 0.66s average response time (≤ 0.5s for Ludicrous Speed Max accounts), 99.95% SLAs, pay for successful responses only.
<a href="https://serpapi.com/" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/serpapi.png" alt="SerpAPI" height="67" width="63"></a>

------

### Bypass Cloudflare with API
Yescaptcha is a proxy service that bypasses Cloudflare and uses the API interface to
obtain verified cookies (e.g. `cf_clearance`). Click [here](https://yescaptcha.com/i/stfnIO)
to register: https://yescaptcha.com/i/stfnIO
<a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>

------

### Easy Captcha Bypass for Scraping
[CapSolver](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC)
is an AI-powered tool that easily bypasses Captchas, allowing uninterrupted access to
public data. It supports a variety of Captchas and works seamlessly with `curl_cffi`,
Puppeteer, Playwright, and more. Fast, reliable, and cost-effective. Plus, `curl_cffi`
users can use the code **"CURL"** to get an extra 6% balance! and register
[here](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC).

<a href="https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/capsolver.jpg" alt="CapSolver" height="50" width="178"></a>

------

## Quick Comparison

| Feature       | requests | aiohttp | httpx | pycurl | curl_cffi |
| :------------ | :------- | :------ | :---- | :----- | :-------- |
| HTTP/2        | ❌       | ❌      | ✅     | ✅     | ✅        |
| HTTP/3        | ❌       | ❌      | ❌    | ☑️<sup>1</sup> | ✅<sup>2</sup>   |
| Sync          | ✅       | ❌      | ✅     | ✅     | ✅        |
| Async         | ❌       | ✅      | ✅     | ❌     | ✅        |
| Websocket     | ❌       | ✅      | ❌    | ❌     | ✅        |
| Fingerprints  | ❌       | ❌      | ❌    | ❌     | ✅        |
| Speed         | 🐇       | 🐇🐇     | 🐇    | 🐇🐇    | 🐇🐇      |

*   Notes:
    1.  Requires HTTP/3 enabled libcurl.
    2.  Available since v0.11.4.

## Installation

Install `curl_cffi` using pip:

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

To install unstable version from GitHub:

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

On macOS, ensure you have dependencies installed:

```bash
brew install zstd nghttp2
```

## Usage

`curl_cffi` offers both a low-level `curl` API and a high-level `requests`-like API.

### `requests`-like API (v0.10+)

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")

print(r.json())

# Impersonate and specify version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Randomly choose a browser version:
r = curl_cffi.get("https://example.com", impersonate="realworld")

# Using proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions (v0.10+)

```python
s = curl_cffi.Session()

s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)

r = s.get("https://httpbin.org/cookies")
print(r.json())
```

### Supported Browsers

`curl_cffi` supports browser versions matching the `curl-impersonate` fork:

| Browser         | Open Source                                                                                                          | Pro Version                                                                                                                                          |
| :-------------- | :------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chrome          | chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116<sup>[1]</sup>, chrome119<sup>[1]</sup>, chrome120<sup>[1]</sup>, chrome123<sup>[3]</sup>, chrome124<sup>[3]</sup>, chrome131<sup>[4]</sup>, chrome133a<sup>[5][6]</sup>, chrome136<sup>[6]</sup> | chrome132, chrome134, chrome135                                                                                                                      |
| Chrome Android  | chrome99_android, chrome131_android <sup>[4]</sup>                                                                      | chrome132_android, chrome133_android, chrome134_android, chrome135_android                                                                           |
| Chrome iOS      | N/A                                                                                                                  | coming soon                                                                                                                                          |
| Safari          | safari153 <sup>[2]</sup>, safari155 <sup>[2]</sup>, safari170 <sup>[1]</sup>, safari180 <sup>[4]</sup>, safari184 <sup>[6]</sup>, safari260 <sup>[8]</sup>                                                                                             | coming soon                                                                                                                                          |
| Safari iOS      | safari172_ios<sup>[1]</sup>, safari180_ios<sup>[4]</sup>, safari184_ios <sup>[6]</sup>, safari260_ios <sup>[8]</sup>                                                                                             | coming soon                                                                                                                                          |
| Firefox         | firefox133<sup>[5]</sup>, firefox135<sup>[7]</sup>                                                                                                                                               | coming soon                                                                                                                                          |
| Firefox Android | N/A                                                                                                                  | firefox135_android                                                                                                                                    |
| Tor             | tor145 <sup>[7]</sup>                                                                                                  | coming soon                                                                                                                                          |
| Edge            | edge99, edge101                                                                                                       | edge133, edge135                                                                                                                                    |
| Opera           | N/A                                                                                                                  | coming soon                                                                                                                                          |
| Brave           | N/A                                                                                                                  | coming soon                                                                                                                                          |

*   Notes:
    1.  Added in version `0.6.0`.
    2.  Fixed in version `0.6.0`, previous http2 fingerprints were [not correct](https://github.com/lwthiker/curl-impersonate/issues/215).
    3.  Added in version `0.7.0`.
    4.  Added in version `0.8.0`.
    5.  Added in version `0.9.0`.
    6.  The version postfix `-a`(e.g. `chrome133a`) means that this is an alternative version, i.e. the fingerprint has not been officially updated by browser, but has been observed because of A/B testing.
    7.  Added in version `0.10.0`.
    8.  Added in version `0.11.0`.
    9.  Since `0.11.0`, the format `safari184_ios` is preferred over `safari18_4_ios`, both are supported, but the latter is quite confusing and hard to parse.
    10. Added in `0.12.0`.

### Asyncio

```python
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

### Asyncio - More concurrency

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

*   Scrapy: [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   Adapters: [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) (requests adapter), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) (httpx adapter).
*   Captcha Resolvers: [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Acknowledgements

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi), licensed under MIT.
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py), licensed under BSD.
*   Asyncio support is inspired by Tornado's curl http client.
*   Synchronous WebSocket API inspired by [websocket\_client](https://github.com/websocket-client/websocket-client).
*   Asynchronous WebSocket API inspired by [aiohttp](https://github.com/aio-libs/aiohttp).