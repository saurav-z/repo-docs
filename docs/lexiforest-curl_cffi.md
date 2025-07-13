# curl_cffi: Python Binding for Advanced Web Scraping and Impersonation

**Bypass website restrictions and scrape the web like a browser with `curl_cffi`, the fastest and most versatile Python binding for `curl`!** ([Original Repository](https://github.com/lexiforest/curl_cffi))

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

`curl_cffi` provides a powerful and efficient way to make HTTP requests in Python, excelling at impersonating browsers and bypassing anti-scraping measures. Built on the robust `curl-impersonate` fork, it offers advanced features for developers needing to scrape data or interact with websites that employ sophisticated bot detection.

[Documentation](https://curl-cffi.readthedocs.io) | [Commercial Support](https://impersonate.pro)

## Key Features

*   **Browser Impersonation:**  Mimics browser TLS/JA3 and HTTP/2 fingerprints, including support for recent browser versions and custom fingerprints.
*   **High Performance:** Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl`.
*   **Familiar API:**  Offers a user-friendly `requests`-like API, minimizing the learning curve.
*   **Asynchronous Support:**  Includes built-in `asyncio` support with proxy rotation for concurrent requests.
*   **HTTP/2 & HTTP/3 Support:**  Provides native support for both HTTP/2 and HTTP/3.
*   **Websocket Support:** Supports both synchronous and asynchronous WebSockets.
*   **Easy Installation:** Pre-compiled wheels for easy installation on Linux, macOS, and Windows.
*   **MIT Licensed:** Open-source and freely available.

## Why Choose curl_cffi?

*   **Bypass Anti-Scraping:**  Effectively bypasses website restrictions by impersonating real browsers.
*   **Superior Speed:** Delivers exceptional performance compared to other Python HTTP clients.
*   **Flexible and Versatile:** Supports a wide range of features, including proxies, and custom fingerprints.
*   **Comprehensive Support:** Offers both open-source and commercial support options.

## Installation

Install `curl_cffi` using pip:

```bash
pip install curl_cffi --upgrade
```

## Usage Examples

###  Requests-like API (v0.10+)
```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Impersonate a specific version (e.g. Chrome 124)
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")
```

### Asyncio

```python
from curl_cffi import AsyncSession
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com")
    print(r.status_code)
```

## Ecosystem Integrations

*   **Scrapy:**  `scrapy-curl-cffi`, `scrapy-impersonate`, and `scrapy-fingerprint` for seamless integration with Scrapy.
*   **Adapters:** Integrate with existing libraries such as [requests](https://github.com/el1s7/curl-adapter) and [httpx](https://github.com/vgavro/httpx-curl-cffi)
*   **Captcha Resolvers:** Compatible with CapSolver and YesCaptcha for automated captcha solving.

## Sponsors

Maintenance of this project is made possible by the <a href="https://github.com/lexiforest/curl_cffi/graphs/contributors">contributors</a> and <a href="https://github.com/sponsors/lexiforest">sponsors</a>.

------

## Integrations and Promotions

*   **SerpApi:**  Scrape search engines with a fast and reliable API.

    <a href="https://serpapi.com/" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/serpapi.png" alt="SerpAPI" height="67" width="63"></a>

------
*   **Yescaptcha:**  Bypass Cloudflare with ease.

    <a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>

------
*   **CapSolver:**  AI-powered Captcha solving.

    <a href="https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/capsolver.jpg" alt="CapSolver" height="50" width="178"></a>

**Use code "CURL" for a 6% balance bonus on CapSolver!**  [Register Here](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC)

## Acknowledgements
*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py).
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).