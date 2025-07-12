# curl_cffi: Mimic Browser Fingerprints for Advanced Web Scraping

Tired of being blocked? **`curl_cffi` is a powerful Python library that lets you bypass anti-bot measures by impersonating real web browsers, providing unparalleled control over your HTTP requests.**  ([Original Repository](https://github.com/lexiforest/curl_cffi))

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Generic badge](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Generic badge](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)
[Documentation](https://curl-cffi.readthedocs.io)

Built upon a custom fork of `curl-impersonate` via `cffi`, `curl_cffi` allows you to send HTTP requests with the same TLS/JA3 and HTTP/2 fingerprints as popular web browsers, giving you a significant advantage when scraping or interacting with websites that employ bot detection. For commercial support, visit [impersonate.pro](https://impersonate.pro).

## Key Features

*   **Browser Impersonation:**  Mimics the fingerprints (TLS/JA3, HTTP/2) of various browsers (Chrome, Safari, Firefox, etc.) to avoid detection.
*   **High Performance:** Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl`.
*   **Familiar API:**  Uses a `requests`-like API for easy integration.
*   **Asynchronous Support:** Full `asyncio` support with proxy rotation capabilities.
*   **HTTP/2 & HTTP/3 Support:**  Includes support for modern HTTP protocols (2.0 & 3.0).
*   **WebSocket Support:**  Includes both synchronous and asynchronous WebSocket APIs.
*   **Easy Installation:**  Pre-compiled wheels for Linux, macOS, and Windows, simplifying setup.
*   **Customization:**  Ability to specify custom JA3/Akamai fingerprints.
*   **MIT Licensed:** Open-source and freely available for use.

## Performance Comparison
| Feature        | requests | aiohttp | httpx   | pycurl  | curl_cffi |
|----------------|----------|---------|---------|---------|-----------|
| HTTP/2         | ❌       | ❌      | ✅      | ✅      | ✅        |
| HTTP/3         | ❌       | ❌      | ❌      | ☑️<sup>1</sup>| ✅<sup>2</sup> |
| Sync           | ✅       | ❌      | ✅      | ✅      | ✅        |
| Async          | ❌       | ✅      | ✅      | ❌      | ✅        |
| WebSocket      | ❌       | ✅      | ❌      | ❌      | ✅        |
| Fingerprints   | ❌       | ❌      | ❌      | ❌      | ✅        |
| Speed          | 🐇       | 🐇🐇      | 🐇      | 🐇🐇      | 🐇🐇      |

Notes:
1. For pycurl, you need an http/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2. Since v0.11.4.


## Installation

```bash
pip install curl_cffi --upgrade
```

## Usage

`curl_cffi` provides both a high-level, `requests`-like API and a low-level `curl` API.

###  requests-like API

```python
import curl_cffi

# Impersonate Chrome (latest version)
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")

print(r.json())

# Impersonate a specific browser version
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
```

### Supported Browser Versions

`curl_cffi` supports the same browser versions as supported by my [fork](https://github.com/lexiforest/curl-impersonate) of [curl-impersonate](https://github.com/lwthiker/curl-impersonate):

Open source version of curl_cffi includes versions whose fingerprints differ from previous versions.
If you see a version, e.g. `chrome135`, were skipped, you can simply impersonate it with your own headers and the previous version.

If you don't want to look up the headers etc, by yourself, consider buying commercial support from [impersonate.pro](https://impersonate.pro),
we have comprehensive browser fingerprints database for almost all the browser versions on various platforms.

If you are trying to impersonate a target other than a browser, use `ja3=...` and `akamai=...`
to specify your own customized fingerprints. See the [docs on impersonation](https://curl-cffi.readthedocs.io/en/latest/impersonate.html) for details.

|Browser|Open Source| Pro version|
|---|---|---|
|Chrome|chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116<sup>[1]</sup>, chrome119<sup>[1]</sup>, chrome120<sup>[1]</sup>, chrome123<sup>[3]</sup>, chrome124<sup>[3]</sup>, chrome131<sup>[4]</sup>, chrome133a<sup>[5][6]</sup>, chrome136<sup>[6]</sup>|chrome132, chrome134, chrome135|
|Chrome Android| chrome99_android, chrome131_android <sup>[4]</sup>|chrome132_android, chrome133_android, chrome134_android, chrome135_android|
|Chrome iOS|N/A|coming soon|
|Safari <sup>[7]</sup>|safari153 <sup>[2]</sup>, safari155 <sup>[2]</sup>, safari170 <sup>[1]</sup>, safari180 <sup>[4]</sup>, safari184 <sup>[6]</sup>, safari260 <sup>[8]</sup>|coming soon|
|Safari iOS <sup>[7]</sup>| safari172_ios<sup>[1]</sup>, safari180_ios<sup>[4]</sup>, safari184_ios <sup>[6]</sup>, safari260_ios <sup>[8]</sup>|coming soon|
|Firefox|firefox133<sup>[5]</sup>, firefox135<sup>[7]</sup>|coming soon|
|Firefox Android|N/A|firefox135_android|
|Tor|tor145 <sup>[7]</sup>|coming soon|
|Edge|edge99, edge101|edge133, edge135|
|Opera|N/A|coming soon|
|Brave|N/A|coming soon|


Notes:
1. Added in version `0.6.0`.
2. Fixed in version `0.6.0`, previous http2 fingerprints were [not correct](https://github.com/lwthiker/curl-impersonate/issues/215).
3. Added in version `0.7.0`.
4. Added in version `0.8.0`.
5. Added in version `0.9.0`.
6. The version postfix `-a`(e.g. `chrome133a`) means that this is an alternative version, i.e. the fingerprint has not been officially updated by browser, but has been observed because of A/B testing.
5. Added in version `0.10.0`.
6. Added in version `0.11.0`.
7. Since `0.11.0`, the format `safari184_ios` is preferred over `safari18_4_ios`, both are supported, but the latter is quite confusing and hard to parse.
8. Added in  `0.12.0`.

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

`curl_cffi` seamlessly integrates with popular scraping and automation tools:

*   **Scrapy:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapters:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) and [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) as adapters.
*   **Captcha Resolvers:** Integrations with services like [CapSolver](https://docs.capsolver.com/en/api/) and [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

------

## Sponsors

Maintenance of this project is made possible by all the <a href="https://github.com/lexiforest/curl_cffi/graphs/contributors">contributors</a> and <a href="https://github.com/sponsors/lexiforest">sponsors</a>. If you'd like to sponsor this project and have your avatar or company logo appear below <a href="https://github.com/sponsors/lexiforest">click here</a>. 💖

------

<a href="https://serpapi.com/" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/serpapi.png" alt="SerpAPI" height="67" width="63"></a>

Scrape Google and other search engines from [SerpApi](https://serpapi.com/)'s fast, easy, and complete API. 0.66s average response time (≤ 0.5s for Ludicrous Speed Max accounts), 99.95% SLAs, pay for successful responses only.

------

### Bypass Cloudflare with API

<a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>

Yescaptcha is a proxy service that bypasses Cloudflare and uses the API interface to
obtain verified cookies (e.g. `cf_clearance`). Click [here](https://yescaptcha.com/i/stfnIO)
to register: https://yescaptcha.com/i/stfnIO

------

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

*   Forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi) (MIT License).
*   Headers/Cookies files from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py) (BSD License).
*   Asyncio support inspired by Tornado's curl HTTP client.
*   Synchronous WebSocket API inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   Asynchronous WebSocket API inspired by [aiohttp](https://github.com/aio-libs/aiohttp).