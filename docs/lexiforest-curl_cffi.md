# curl_cffi: The Ultimate Python Library for Web Scraping and Browser Impersonation

**Tired of getting blocked?** `curl_cffi` is a powerful Python library that allows you to bypass anti-bot measures and web scraping protections by mimicking real browser fingerprints, making your web requests undetectable.  [Check out the original repo](https://github.com/lexiforest/curl_cffi)!

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

## Key Features

*   **Browser Impersonation:**  Mimic various browsers (Chrome, Safari, Firefox, Edge, and others) with accurate TLS/JA3 and HTTP/2 fingerprints, including specific versions.
*   **High Performance:**  Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`.
*   **Familiar API:**  Uses a `requests`-like API for ease of use.
*   **Asynchronous Support:**  Full `asyncio` integration with proxy rotation for efficient scraping.
*   **HTTP/2 & HTTP/3 Support:** Includes support for modern protocols, unlike the `requests` library.
*   **Websocket Support:** Seamless integration with WebSockets for real-time data.
*   **Pre-compiled and Easy Installation:**  Ready to use out-of-the-box on most platforms.
*   **Flexible Proxy Support:**  Works with both HTTP/SOCKS proxies.
*   **Bypass Cloudflare and Captchas:**  Integrate with captcha solving services like YesCaptcha.

## Why Choose curl_cffi?

`curl_cffi` is the leading Python binding for `curl`, designed to bypass web scraping protections. With its ability to impersonate browser fingerprints, it allows you to access websites that might block standard HTTP clients. Compared to other HTTP clients, it provides advanced features like HTTP/3 and WebSockets while also being much faster.

## Installation

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

# Specify a browser version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Use proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Asynchronous Usage

```python
from curl_cffi import AsyncSession
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com", impersonate="chrome")
    print(r.text)
```

## Browser Support

`curl_cffi` supports a wide range of browser versions, with detailed information available in the documentation.
The open source version includes versions whose fingerprints differ from previous versions.

If you need advanced support like specific versions, consider commercial support from [impersonate.pro](https://impersonate.pro).

## Ecosystem Integrations

*   Scrapy:  [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint)
*   Adapters:  [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) for `requests`, [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) for `httpx`
*   Captcha Resolvers:  [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview)

## Sponsors

Maintenance of this project is made possible by all the <a href="https://github.com/lexiforest/curl_cffi/graphs/contributors">contributors</a> and <a href="https://github.com/sponsors/lexiforest">sponsors</a>. If you'd like to sponsor this project and have your avatar or company logo appear below <a href="https://github.com/sponsors/lexiforest">click here</a>. 💖

## Contributing

Contributions are welcome! Please submit pull requests on a different branch than `main` and check the "Allow edits by maintainers" box for ease of merging.