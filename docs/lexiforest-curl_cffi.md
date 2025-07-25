# curl_cffi: The Python Library That Bypasses Website Blocks

**`curl_cffi` is a powerful Python library built for high-performance HTTP requests, enabling you to bypass website restrictions by impersonating browser fingerprints.** This allows you to scrape data and access content that might otherwise be blocked.

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi.svg)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io) | [GitHub Repository](https://github.com/lexiforest/curl_cffi)

`curl_cffi` leverages the `curl-impersonate` fork and `cffi` to provide a high-speed and flexible HTTP client. Unlike other Python HTTP clients, `curl_cffi` can mimic the fingerprints of various browsers, including TLS/JA3 and HTTP/2, making it ideal for web scraping and bypassing bot detection.

**Key Features:**

*   **Browser Impersonation:** Mimics browser fingerprints (TLS/JA3 and HTTP/2) for Chrome, Safari, Firefox, and more, allowing you to access websites that may block other clients.
*   **High Performance:**  Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`. See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API, making it easy to learn and use.
*   **Asynchronous Support:**  Supports `asyncio` with proxy rotation on each request for efficient concurrent operations.
*   **HTTP/2 & HTTP/3 Support:** Offers built-in support for modern HTTP protocols.
*   **Websocket Support:** Includes support for both synchronous and asynchronous WebSockets.
*   **Pre-compiled:**  No need to compile on your machine, making installation straightforward.

### Ecosystem Integrations
*   **Scrapy Integration:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapters:** [requests](https://github.com/el1s7/curl-adapter), [httpx](https://github.com/vgavro/httpx-curl-cffi).
*   **Captcha Resolvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Installation

```bash
pip install curl_cffi --upgrade
```

*   Supports Linux, macOS, and Windows out of the box.
*   Beta releases: `pip install curl_cffi --upgrade --pre`
*   Unstable versions: [See the GitHub Repo](https://github.com/lexiforest/curl_cffi)

### Dependencies (macOS)

```bash
brew install zstd nghttp2
```

## Usage

`curl_cffi` offers both low-level `curl` APIs and a high-level `requests`-like API.

### `requests`-like API

```python
import curl_cffi

# Notice the impersonate parameter
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")

print(r.json())
# output: {..., "ja3n_hash": "aa56c057ad164ec4fdcb7a5a283be9fc", ...}
# the js3n fingerprint should be the same as target browser

# To keep using the latest browser version as `curl_cffi` updates,
# simply set impersonate="chrome" without specifying a version.
# Other similar values are: "safari" and "safari_ios"
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")

# Randomly choose a browser version based on current market share in real world
# from: https://caniuse.com/usage-table
# NOTE: this is a pro feature.
r = curl_cffi.get("https://example.com", impersonate="realworld")

# To pin a specific version, use version numbers together.
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# To impersonate other than browsers, bring your own ja3/akamai strings
# See examples directory for details.
r = curl_cffi.get("https://tls.browserleaks.com/json", ja3=..., akamai=...)

# http/socks proxies are supported
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)

proxies = {"https": "socks://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()

# httpbin is a http test website, this endpoint makes the server set cookies
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
# <Cookies[<Cookie foo=bar for httpbin.org />]>

# retrieve cookies again to verify
r = s.get("https://httpbin.org/cookies")
print(r.json())
# {'cookies': {'foo': 'bar'}}
```

### Supported Browsers and Versions

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

### Asyncio

```python
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

More concurrency:

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

For low-level APIs, Scrapy integration and other advanced topics, see the
[docs](https://curl-cffi.readthedocs.io) for more details.

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

## Acknowledgements

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi), which is under the MIT license.
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py), which is under the BSD license.
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).

## Sponsors
[Sponsor Me](https://github.com/sponsors/lexiforest)