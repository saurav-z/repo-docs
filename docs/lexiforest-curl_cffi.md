# curl_cffi: Blazing-Fast Python HTTP Client with Browser Impersonation

**Bypass website restrictions and access the web like a browser with `curl_cffi`, the fastest Python HTTP client available.** [Check out the original repo!](https://github.com/lexiforest/curl_cffi)

[![PyPI Downloads](https://static.pepy.tech/badge/curl-cffi/week)](https://pepy.tech/projects/curl-cffi)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://img.shields.io/pypi/pyversions/curl_cffi)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl_cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` is a Python binding for the `curl-impersonate` fork, offering powerful features for web scraping, data extraction, and more. Unlike other Python HTTP clients, `curl_cffi` excels at mimicking browser behavior, including TLS/JA3 and HTTP/2 fingerprints, making it ideal for bypassing anti-scraping measures.

## Key Features

*   **Browser Impersonation:** Emulates popular browsers (Chrome, Safari, Firefox, etc.) to avoid detection.
*   **Blazing Fast Performance:** Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl`.
*   **Requests-like API:** Easy to learn and use, familiar to users of the `requests` library.
*   **Pre-compiled:** No need to compile on your machine; ready to use out of the box.
*   **Asynchronous Support:** Full `asyncio` support with proxy rotation for efficient web scraping.
*   **HTTP/2 & HTTP/3 Support:** Includes support for modern HTTP protocols.
*   **WebSockets:** Supports both synchronous and asynchronous WebSockets.
*   **MIT Licensed:** Permissive license for flexible use.

## Performance Comparison

| Feature       | requests | aiohttp | httpx | pycurl | curl_cffi |
|---------------|----------|---------|-------|--------|-----------|
| HTTP/2        | ❌       | ❌      | ✅    | ✅     | ✅        |
| HTTP/3        | ❌       | ❌      | ❌    | ☑️<sup>1</sup> | ✅<sup>2</sup>   |
| Sync          | ✅       | ❌      | ✅    | ✅     | ✅        |
| Async         | ❌       | ✅      | ✅    | ❌     | ✅        |
| WebSockets    | ❌       | ✅      | ❌    | ❌     | ✅        |
| Fingerprints  | ❌       | ❌      | ❌    | ❌     | ✅        |
| Speed         | 🐇       | 🐇🐇    | 🐇    | 🐇🐇   | 🐇🐇      |

Notes:
1. For pycurl, you need an http/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2. Since v0.11.4.

## Installation

Install `curl_cffi` using pip:

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

## Usage Examples

### Requests-like API

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Pin a specific Chrome version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Use proxies
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

### Asynchronous Requests

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

## Supported Browser Versions

`curl_cffi` supports the same browser versions as the  [curl-impersonate](https://github.com/lwthiker/curl-impersonate) project:

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

*For comprehensive browser fingerprint support, consider commercial support from [impersonate.pro](https://impersonate.pro).*

## Ecosystem

*   **Scrapy Integration:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapter for requests:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter)
*   **Adapter for httpx:** [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi)
*   **Captcha Solvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Acknowledgements

*   Forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
*   Headers/Cookies files from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py).
*   Asyncio support inspired by Tornado's curl HTTP client.
*   Synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   Asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).

## Contributing

Contributions are welcome! Please use a separate branch and enable "Allow edits by maintainers" when submitting a pull request.