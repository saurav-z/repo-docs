# curl_cffi:  The Fastest Python HTTP Client for Browser Impersonation

Tired of getting blocked? **curl_cffi** is a powerful Python library that lets you effortlessly mimic real browser behavior, bypassing website restrictions and accessing content that's otherwise unavailable.  Check out the [original repo](https://github.com/lexiforest/curl_cffi) for more details.

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)
[Documentation](https://curl-cffi.readthedocs.io)

## Key Features of curl_cffi:

*   **Browser Impersonation:** Mimic the TLS/JA3 and HTTP/2 fingerprints of modern browsers like Chrome, Safari, and Firefox.
*   **Blazing Fast Performance:** Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl`. See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API for easy adoption.
*   **Pre-compiled:** No need to compile on your machine; ready to use out of the box.
*   **Asynchronous Support:** Includes `asyncio` support with proxy rotation for asynchronous operations.
*   **HTTP/2 & HTTP/3 Support:**  Leverages modern protocols for improved performance.
*   **Websocket Support:**  Supports both synchronous and asynchronous WebSockets.
*   **MIT Licensed:** Use freely in your projects.

## Why Choose curl_cffi?

`curl_cffi` provides a robust and efficient solution for web scraping, data extraction, and bypassing bot detection.  It's the perfect choice when you need to:

*   Access websites that block standard HTTP clients.
*   Achieve high scraping speeds without sacrificing accuracy.
*   Easily integrate browser-like behavior into your Python projects.

## Comparison: curl_cffi vs. Other HTTP Clients

| Feature           | requests | aiohttp | httpx  | pycurl | curl_cffi |
|-------------------|----------|---------|--------|--------|-----------|
| HTTP/2            | ❌       | ❌      | ✅     | ✅     | ✅        |
| HTTP/3            | ❌       | ❌      | ❌     | ☑️<sup>1</sup> | ✅<sup>2</sup> |
| Synchronous       | ✅       | ❌      | ✅     | ✅     | ✅        |
| Asynchronous      | ❌       | ✅      | ✅     | ❌     | ✅        |
| WebSockets        | ❌       | ✅      | ❌     | ❌     | ✅        |
| Browser Fingerprints | ❌       | ❌      | ❌     | ❌     | ✅        |
| Speed             | 🐇       | 🐇🐇    | 🐇     | 🐇🐇   | 🐇🐇      |

Notes:
1.  For pycurl, you need an http/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2.  Since v0.11.4.

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

# Impersonate with a specific Chrome version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Use a proxy
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
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com", impersonate="chrome")
    print(r.text)
```

```python
import asyncio
from curl_cffi import AsyncSession

async def fetch(url):
    async with AsyncSession() as s:
        r = await s.get(url)
        return r.status_code

async def main():
    urls = ["https://www.example.com", "https://httpbin.org/get", "https://www.python.org"]
    tasks = [fetch(url) for url in urls]
    results = await asyncio.gather(*tasks)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
```

## Supported Browser Versions

`curl_cffi` supports various browser versions, including:

*   Chrome
*   Chrome Android
*   Safari
*   Safari iOS
*   Firefox
*   Firefox Android
*   Edge
*   Tor

For the latest supported versions and details, refer to the [documentation](https://curl-cffi.readthedocs.io/en/latest/impersonate.html).

## Ecosystem Integrations

`curl_cffi` seamlessly integrates with popular tools:

*   **Scrapy:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Requests & httpx Adapters:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi).
*   **Captcha Solvers:**  Compatible with services like CapSolver and YesCaptcha.

## Contributing

Contributions are welcome!  Please submit pull requests from a separate branch and check the "Allow edits by maintainers" box for easier integration.

## Acknowledgements

This project builds upon the work of:

*   [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi)
*   [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py)
*   Tornado, websocket_client, aiohttp