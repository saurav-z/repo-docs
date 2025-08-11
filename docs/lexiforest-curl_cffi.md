# curl_cffi: The Fastest Python HTTP Client with Browser Impersonation

**Bypass website restrictions and achieve superior web scraping with `curl_cffi`, the Python library that lets you impersonate browsers and leverage the power of `curl`.** ([Original Repository](https://github.com/lexiforest/curl_cffi))

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)
[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` provides a high-performance Python binding for `curl`, built upon the `curl-impersonate` fork and `cffi`. It excels at bypassing website restrictions through advanced browser fingerprinting and efficient HTTP/2 & HTTP/3 support. For commercial support, visit [impersonate.pro](https://impersonate.pro).

## Key Features

*   **Browser Impersonation:** Mimics browser TLS/JA3 and HTTP/2 fingerprints, enabling you to bypass bot detection.
*   **High Performance:** Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl`.  See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API for ease of use.
*   **Pre-compiled:** No compilation needed on your machine.
*   **Asyncio Support:** Includes asynchronous support with proxy rotation for efficient web scraping.
*   **HTTP/2 & HTTP/3 Support:** Supports modern protocols not available in some other libraries.
*   **WebSockets Support:** Provides robust WebSocket client capabilities.
*   **MIT Licensed:**  Free to use and integrate into your projects.

## Performance Comparison

| Feature       | requests | aiohttp | httpx | pycurl | curl_cffi |
|---------------|----------|---------|-------|--------|-----------|
| HTTP/2        | ❌       | ❌      | ✅    | ✅     | ✅        |
| HTTP/3        | ❌       | ❌      | ❌    | ☑️<sup>1</sup>    | ✅<sup>2</sup>     |
| Sync          | ✅       | ❌      | ✅    | ✅     | ✅        |
| Async         | ❌       | ✅      | ✅    | ❌     | ✅        |
| Websocket     | ❌       | ✅      | ❌    | ❌     | ✅        |
| Fingerprints  | ❌       | ❌      | ❌    | ❌     | ✅        |
| Speed         | 🐇       | 🐇🐇     | 🐇    | 🐇🐇   | 🐇🐇      |

*Notes:*
1.  For `pycurl`, you need an http/3 enabled libcurl.
2.  Since v0.11.4.

## Installation

```bash
pip install curl_cffi --upgrade
```

## Usage Examples

```python
import curl_cffi

# Requests-like API with browser impersonation
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Asyncio Example
from curl_cffi import AsyncSession
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com", impersonate="chrome")
    print(r.text)
```

## Supported Browsers

`curl_cffi` supports a wide range of browser versions for impersonation, updated regularly.  See the original README for the most current list.

## Ecosystem Integration

*   **Scrapy:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapters:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) and [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi).
*   **Captcha Solvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Contributing

Contributions are welcome!  Please submit pull requests from branches other than `main` and check "Allow edits by maintainers" to facilitate the process.