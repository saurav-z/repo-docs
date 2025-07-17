# curl_cffi: Python HTTP Client for Advanced Web Scraping and Impersonation

**Bypass anti-bot systems and scrape websites effectively with `curl_cffi`, a Python binding for `curl` that allows you to impersonate browsers and customize HTTP requests.**  [View the original repo on GitHub](https://github.com/lexiforest/curl_cffi)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)
[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` is a powerful Python library built on top of `curl-impersonate`, offering a flexible and performant way to make HTTP requests. It excels where other HTTP clients fall short, enabling you to:

**Key Features:**

*   **Browser Impersonation:** Mimic the behavior of various browsers (Chrome, Safari, Firefox, etc.) and their specific versions, including TLS/JA3 and HTTP/2 fingerprints, to bypass bot detection.
*   **High Performance:** Outperforms `requests` and often matches the speed of `aiohttp` and `pycurl`, making it ideal for high-volume scraping. See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:**  Offers a `requests`-like API for ease of use, reducing the learning curve.
*   **Asynchronous Support:** Includes robust `asyncio` support, allowing for efficient concurrent requests.
*   **HTTP/2 and HTTP/3 Compatibility:** Supports the latest HTTP protocols, which are often required by modern websites.
*   **Websocket Support:**  Supports Websocket for real-time data streaming.
*   **Proxy Support:**  Supports HTTP and SOCKS proxies with proxy rotation capabilities in async mode.
*   **Pre-compiled:** Offers pre-compiled wheels for easy installation and use across multiple platforms.

## Installation

Install `curl_cffi` using pip:

```bash
pip install curl_cffi --upgrade
```

## Usage

### Requests-like API

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Pin a specific version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")
```

### Sessions

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
```

### Asyncio

```python
from curl_cffi import AsyncSession
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com")
    print(r.text)
```

## Ecosystem & Integrations

`curl_cffi` seamlessly integrates with other popular libraries and services:

*   **Scrapy:** Use `curl_cffi` within your Scrapy spiders for more effective scraping:  [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Requests & HTTTPX Adapters:** Extend `requests` and `httpx` capabilities with `curl_cffi` adapters: [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi).
*   **Captcha Resolvers:** Integrate with captcha solving services for automated scraping: [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Sponsors

[View Contributors and Sponsors](https://github.com/lexiforest/curl_cffi/graphs/contributors)

## Commercial Support

For advanced browser fingerprinting, comprehensive support, and custom solutions, visit [impersonate.pro](https://impersonate.pro).

***

**Note:** This README is derived from the original provided content and includes SEO-optimized keywords to help with search engine discoverability.