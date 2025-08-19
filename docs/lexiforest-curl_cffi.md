# curl_cffi: The Fastest Python HTTP Client that Impersonates Browsers

**Tired of being blocked?**  `curl_cffi` is a high-performance Python library that lets you effortlessly mimic browser fingerprints to bypass anti-bot defenses and access websites without restrictions.  [Check out the original repo for more details](https://github.com/lexiforest/curl_cffi).

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io) | [Commercial Support](https://impersonate.pro)

**Key Features:**

*   ✅ **Browser Impersonation:**  Mimics TLS/JA3 and HTTP/2 fingerprints of popular browsers (Chrome, Safari, Firefox, Edge, and more) and supports custom fingerprints.
*   🚀 **Blazing Fast:** Significantly faster than `requests` and `httpx`, on par with `aiohttp` and `pycurl`.
*   💻 **Familiar API:**  Uses a `requests`-like API for easy integration.
*   📦 **Pre-compiled:** No need to compile on your machine.
*   🔄 **Asyncio Support:**  Includes `asyncio` support with proxy rotation for each request.
*   🌐 **Modern Protocol Support:** Supports HTTP/2 and HTTP/3.
*   🕸️ **Websocket Support:** Includes both synchronous and asynchronous WebSocket APIs.
*   🛡️ **MIT License:**  Use it freely.

### Bypass Cloudflare with API

<a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>

Yescaptcha is a proxy service that bypasses Cloudflare and uses the API interface to obtain verified cookies (e.g. `cf_clearance`). Click [here](https://yescaptcha.com/i/stfnIO) to register.

------

## Why Choose curl_cffi?

`curl_cffi` offers a significant performance boost and advanced features compared to other Python HTTP clients, especially when dealing with websites that employ anti-bot measures.

| Feature          | requests | aiohttp | httpx   | pycurl   | curl_cffi |
| ---------------- | -------- | ------- | ------- | -------- | ----------- |
| HTTP/2           | ❌       | ❌      | ✅      | ✅       | ✅          |
| HTTP/3           | ❌       | ❌      | ❌      | ☑️<sup>1</sup> | ✅<sup>2</sup> |
| Sync             | ✅       | ❌      | ✅      | ✅       | ✅          |
| Async            | ❌       | ✅      | ✅      | ❌       | ✅          |
| Websocket        | ❌       | ✅      | ❌      | ❌       | ✅          |
| Fingerprints     | ❌       | ❌      | ❌      | ❌       | ✅          |
| Speed            | 🐇       | 🐇🐇    | 🐇      | 🐇🐇     | 🐇🐇        |

Notes:
1.  For pycurl, you need an http/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2.  Since v0.11.4.

## Installation

```bash
pip install curl_cffi --upgrade
```

This works out of the box on Linux, macOS, and Windows. If you encounter issues, you may need to install `curl-impersonate` first, and set environment variables like `LD_LIBRARY_PATH`.

**Beta Releases:**

```bash
pip install curl_cffi --upgrade --pre
```

**From GitHub (Unstable):**

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

**macOS Dependencies:**

```bash
brew install zstd nghttp2
```

## Usage Examples

`curl_cffi` provides both a low-level `curl` API and a high-level `requests`-like API:

### Requests-like API

```python
import curl_cffi

r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124") # Specific version
r = curl_cffi.get("https://tls.browserleaks.com/json", ja3=..., akamai=...) # Custom fingerprints
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
r = s.get("https://httpbin.org/cookies")
print(r.json())
```

### Supported Browser Impersonations

`curl_cffi` supports various browser versions, mirroring the capabilities of the [curl-impersonate](https://github.com/lwthiker/curl-impersonate) project. For commercial support and a comprehensive browser fingerprint database, visit [impersonate.pro](https://impersonate.pro).

| Browser         | Open Source                                                                                                                                                                                                                                                      | Pro version                       |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| Chrome          | chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116<sup>[1]</sup>, chrome119<sup>[1]</sup>, chrome120<sup>[1]</sup>, chrome123<sup>[3]</sup>, chrome124<sup>[3]</sup>, chrome131<sup>[4]</sup>, chrome133a<sup>[5][6]</sup>, chrome136<sup>[6]</sup> | chrome132, chrome134, chrome135 |
| Chrome Android  | chrome99_android, chrome131_android <sup>[4]</sup>                                                                                                                                                                                                                | chrome132_android, chrome133_android, chrome134_android, chrome135_android |
| Chrome iOS      | N/A                                                                                                                                                                                                                                                            | coming soon                       |
| Safari          | safari153 <sup>[2]</sup>, safari155 <sup>[2]</sup>, safari170 <sup>[1]</sup>, safari180 <sup>[4]</sup>, safari184 <sup>[6]</sup>, safari260 <sup>[8]</sup>                                                                                                                    | coming soon                       |
| Safari iOS      | safari172_ios<sup>[1]</sup>, safari180_ios<sup>[4]</sup>, safari184_ios <sup>[6]</sup>, safari260_ios <sup>[8]</sup>                                                                                                                                                     | coming soon                       |
| Firefox         | firefox133<sup>[5]</sup>, firefox135<sup>[7]</sup>                                                                                                                                                                                                             | coming soon                       |
| Firefox Android | N/A                                                                                                                                                                                                                                                            | firefox135_android                |
| Tor             | tor145 <sup>[7]</sup>                                                                                                                                                                                                                                             | coming soon                       |
| Edge            | edge99, edge101                                                                                                                                                                                                                                                   | edge133, edge135                  |
| Opera           | N/A                                                                                                                                                                                                                                                            | coming soon                       |
| Brave           | N/A                                                                                                                                                                                                                                                            | coming soon                       |

Notes:
1.  Added in version `0.6.0`.
2.  Fixed in version `0.6.0`, previous http2 fingerprints were [not correct](https://github.com/lwthiker/curl-impersonate/issues/215).
3.  Added in version `0.7.0`.
4.  Added in version `0.8.0`.
5.  Added in version `0.9.0`.
6.  The version postfix `-a`(e.g. `chrome133a`) means that this is an alternative version, i.e. the fingerprint has not been officially updated by browser, but has been observed because of A/B testing.
7.  Added in version `0.10.0`.
8.  Added in version `0.11.0`.
9.  Since `0.11.0`, the format `safari184_ios` is preferred over `safari18_4_ios`, both are supported, but the latter is quite confusing and hard to parse.
10. Added in  `0.12.0`.

### Asyncio Example

```python
from curl_cffi import AsyncSession
async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

### Asyncio - More Concurrency
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

*   **Scrapy:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapters:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) (for `requests`), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) (for `httpx`).
*   **Captcha Resolvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Acknowledgements

*   Forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py).
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).

## Contributing

Please use a separate branch and check "Allow edits by maintainers" when submitting a PR.