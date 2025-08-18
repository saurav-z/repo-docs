# curl_cffi: The Fastest Python HTTP Client That Impersonates Browsers

Tired of getting blocked? **curl_cffi is a high-performance Python library built on `curl-impersonate`, enabling you to seamlessly mimic browser behavior and bypass anti-bot measures.**  Learn more about the original project [here](https://github.com/lexiforest/curl_cffi).

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

## Key Features of curl_cffi

*   **Browser Impersonation:**  Mimics TLS/JA3 and HTTP/2 fingerprints of various browsers (Chrome, Safari, Firefox, etc.) to avoid detection.
*   **High Performance:** Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`, see [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Easy to Use:** Offers a familiar `requests`-like API, minimizing the learning curve.
*   **Asynchronous Support:** Built-in `asyncio` support with proxy rotation for asynchronous tasks.
*   **Modern Protocol Support:** Supports HTTP/2 and HTTP/3, unlike the standard `requests` library.
*   **Websocket Support:** Includes both synchronous and asynchronous WebSocket support.
*   **Pre-compiled:**  No need to compile dependencies on your machine.
*   **Customizable:** Support for JA3 and Akamai fingerprints.

## Install

Install `curl_cffi` easily using pip:

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

## Usage

### requests-like API

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Pin a specific version
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

## Supported Impersonate Browsers

`curl_cffi` supports impersonation of various browsers. For commercial support visit [impersonate.pro](https://impersonate.pro).

|Browser|Open Source|
|---|---|
|Chrome|chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116<sup>[1]</sup>, chrome119<sup>[1]</sup>, chrome120<sup>[1]</sup>, chrome123<sup>[3]</sup>, chrome124<sup>[3]</sup>, chrome131<sup>[4]</sup>, chrome133a<sup>[5][6]</sup>, chrome136<sup>[6]</sup>|
|Chrome Android| chrome99_android, chrome131_android <sup>[4]</sup>|
|Safari <sup>[7]</sup>|safari153 <sup>[2]</sup>, safari155 <sup>[2]</sup>, safari170 <sup>[1]</sup>, safari180 <sup>[4]</sup>, safari184 <sup>[6]</sup>, safari260 <sup>[8]</sup>|
|Safari iOS <sup>[7]</sup>| safari172_ios<sup>[1]</sup>, safari180_ios<sup>[4]</sup>, safari184_ios <sup>[6]</sup>, safari260_ios <sup>[8]</sup>|
|Firefox|firefox133<sup>[5]</sup>, firefox135<sup>[7]</sup>|
|Edge|edge99, edge101|

For a full list of versions see the original README.

## Ecosystem

`curl_cffi` integrates with other projects:

*   Scrapy:  [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   Adaptors: [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) for requests and [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) for httpx
*   Captcha solvers:  [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Contributing

Contributions are welcome! Please create a branch other than `main` and check the "Allow edits by maintainers" box in your pull request.