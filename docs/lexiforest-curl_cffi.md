# curl_cffi: Effortlessly Impersonate Browsers with Python's Fastest HTTP Client

**Bypass website restrictions and achieve lightning-fast web scraping with `curl_cffi`, the Python library that lets you mimic browser behavior.**  [Check out the original repo](https://github.com/lexiforest/curl_cffi) for more details.

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` is a Python binding for a [curl-impersonate fork](https://github.com/lexiforest/curl-impersonate) and uses [cffi](https://cffi.readthedocs.io/en/latest/) to deliver superior performance and browser impersonation capabilities. For commercial support and advanced features, visit [impersonate.pro](https://impersonate.pro).

## Key Features

*   **Browser Impersonation:**  Mimics the TLS/JA3 and HTTP/2 fingerprints of popular browsers (Chrome, Safari, Firefox, Edge, etc.) and custom fingerprints.
*   **Blazing Fast:** Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`, see [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API, so you can get started quickly.
*   **Pre-compiled:** No need to compile anything on your machine, it works out-of-the-box.
*   **Async Support:**  Includes `asyncio` support, with proxy rotation for asynchronous requests.
*   **HTTP/2 & HTTP/3 Compatibility:** Supports modern HTTP/2 and HTTP/3 protocols.
*   **Websocket Support:** Provides built-in support for WebSockets (both synchronous and asynchronous).
*   **MIT License:**  Free to use and integrate into your projects.

### Feature Comparison

| Feature          | requests | aiohttp | httpx | pycurl  | curl_cffi |
| ---------------- | -------- | ------- | ----- | ------- | --------- |
| HTTP/2           | ❌       | ❌      | ✅    | ✅      | ✅        |
| HTTP/3           | ❌       | ❌      | ❌    | ☑️<sup>1</sup> | ✅<sup>2</sup>       |
| Sync             | ✅       | ❌      | ✅    | ✅      | ✅        |
| Async            | ❌       | ✅      | ✅    | ❌      | ✅        |
| WebSocket        | ❌       | ✅      | ❌    | ❌      | ✅        |
| Fingerprints     | ❌       | ❌      | ❌    | ❌      | ✅        |
| Speed            | 🐇       | 🐇🐇     | 🐇    | 🐇🐇     | 🐇🐇       |

**Notes:**

1.  Requires an HTTP/3 enabled `libcurl`.
2.  Available since v0.11.4.

## Installation

Install `curl_cffi` using pip:

```bash
pip install curl_cffi --upgrade
```

## Usage

`curl_cffi` offers both a low-level `curl` API and a higher-level `requests`-like API.

### Requests-like API

```python
import curl_cffi

# Impersonate a browser with `impersonate` parameter
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Use the latest version of a browser
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")

# Randomly select a browser version (Pro Feature)
r = curl_cffi.get("https://example.com", impersonate="realworld")

# Pin a specific browser version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Customize fingerprints with ja3 and akamai
r = curl_cffi.get("https://tls.browserleaks.com/json", ja3=..., akamai=...)

# Use proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)

proxies = {"https": "socks://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()

# Set cookies via a request
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)

# Retrieve cookies
r = s.get("https://httpbin.org/cookies")
print(r.json())
```

### Supported Browsers

`curl_cffi` supports a wide range of browser versions, including:

| Browser         | Open Source                                                                                                                                                                                               | Pro Version                                                                                                                                 |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Chrome          | chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116<sup>[1]</sup>, chrome119<sup>[1]</sup>, chrome120<sup>[1]</sup>, chrome123<sup>[3]</sup>, chrome124<sup>[3]</sup>, chrome131<sup>[4]</sup>, chrome133a<sup>[5][6]</sup>, chrome136<sup>[6]</sup> | chrome132, chrome134, chrome135                                                                                                  |
| Chrome Android  | chrome99_android, chrome131_android <sup>[4]</sup>                                                                                                                                                                  | chrome132_android, chrome133_android, chrome134_android, chrome135_android                                                                 |
| Chrome iOS      | N/A                                                                                                                                                                                                       | coming soon                                                                                                                                  |
| Safari          | safari153 <sup>[2]</sup>, safari155 <sup>[2]</sup>, safari170 <sup>[1]</sup>, safari180 <sup>[4]</sup>, safari184 <sup>[6]</sup>, safari260 <sup>[8]</sup>                                                                                                                                                                                                 | coming soon                                                                                                                                  |
| Safari iOS      | safari172_ios<sup>[1]</sup>, safari180_ios<sup>[4]</sup>, safari184_ios <sup>[6]</sup>, safari260_ios <sup>[8]</sup>                                                                                                                                                                                                 | coming soon                                                                                                                                  |
| Firefox         | firefox133<sup>[5]</sup>, firefox135<sup>[7]</sup>                                                                                                                                                                   | coming soon                                                                                                                                  |
| Firefox Android | N/A                                                                                                                                                                                                       | firefox135_android                                                                                                                          |
| Tor             | tor145 <sup>[7]</sup>                                                                                                                                                                                    | coming soon                                                                                                                                  |
| Edge            | edge99, edge101                                                                                                                                                                                           | edge133, edge135                                                                                                                             |
| Opera           | N/A                                                                                                                                                                                                       | coming soon                                                                                                                                  |
| Brave           | N/A                                                                                                                                                                                                       | coming soon                                                                                                                                  |

**Notes:**

1.  Added in version `0.6.0`.
2.  Fixed in version `0.6.0`, previous http2 fingerprints were [not correct](https://github.com/lwthiker/curl-impersonate/issues/215).
3.  Added in version `0.7.0`.
4.  Added in version `0.8.0`.
5.  Added in version `0.9.0`.
6.  The version postfix `-a`(e.g. `chrome133a`) means that this is an alternative version, i.e. the fingerprint has not been officially updated by browser, but has been observed because of A/B testing.
5.  Added in version `0.10.0`.
6.  Added in version `0.11.0`.
7.  Since `0.11.0`, the format `safari184_ios` is preferred over `safari18_4_ios`, both are supported, but the latter is quite confusing and hard to parse.
8.  Added in  `0.12.0`.

### Asyncio

```python
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

### More Concurrency

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

## Ecosystem & Integrations

`curl_cffi` seamlessly integrates with popular tools and services:

*   **Scrapy:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Requests & httpx Adapters:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi)
*   **Captcha Resolvers:**  [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Sponsors & Acknowledgements

Maintenance of this project is made possible by the contributions of the community and [sponsors](https://github.com/sponsors/lexiforest).  Consider sponsoring to help keep this project running! 💖

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py).
*   Asyncio support inspired by Tornado's curl http client.
*   Synchronous WebSocket API inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   Asynchronous WebSocket API inspired by [aiohttp](https://github.com/aio-libs/aiohttp).

## Contributing

Contributions are welcome!  When submitting a PR, check the "Allow edits by maintainers" box to enable easier integration of linting and style fixes.