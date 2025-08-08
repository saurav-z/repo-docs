# curl_cffi: The Ultimate Python Library for Web Scraping and Browser Impersonation

Tired of being blocked? **curl_cffi** is a powerful Python library that mimics browser behavior, allowing you to bypass bot detection and scrape websites effectively.  [Check out the original repo](https://github.com/lexiforest/curl_cffi).

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)
[Documentation](https://curl-cffi.readthedocs.io)

Built upon the `curl-impersonate` fork and `cffi`, curl_cffi provides a high-performance, browser-mimicking HTTP client. Perfect for web scraping, data extraction, and any task that requires robust and undetectable web requests.

**Key Features:**

*   **Browser Impersonation:** Mimic various browsers' TLS/JA3 and HTTP/2 fingerprints (Chrome, Safari, Firefox, etc.).
*   **High Performance:** Significantly faster than `requests` and `httpx`, on par with `aiohttp` and `pycurl`.
*   **Familiar API:** Uses a `requests`-like API for easy integration.
*   **Asynchronous Support:** Built-in `asyncio` support with proxy rotation.
*   **HTTP/2 & HTTP/3 Support:**  Includes support for the latest HTTP protocols.
*   **Websocket Support:**  Offers both synchronous and asynchronous WebSocket APIs.
*   **Pre-compiled Binaries:** Easy installation without needing to compile on your machine.
*   **Proxy Support:**  Supports HTTP and SOCKS proxies.
*   **Open Source & MIT Licensed:**  Free to use and modify.

**Why Choose curl_cffi?**

| Feature          | requests | aiohttp | httpx | pycurl | curl_cffi |
|------------------|----------|---------|-------|--------|-------------|
| HTTP/2           | ❌       | ❌      | ✅    | ✅     | ✅          |
| HTTP/3           | ❌       | ❌      | ❌    | ✅     | ✅          |
| Sync             | ✅       | ❌      | ✅    | ✅     | ✅          |
| Async            | ❌       | ✅      | ✅    | ❌     | ✅          |
| WebSockets       | ❌       | ✅      | ❌    | ❌     | ✅          |
| Fingerprints     | ❌       | ❌      | ❌    | ❌     | ✅          |
| Speed            | 🐇       | 🐇🐇    | 🐇    | 🐇🐇   | 🐇🐇        |

**Installation**

```bash
pip install curl_cffi --upgrade
```

**Usage**

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Asyncio Example
from curl_cffi import AsyncSession
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com", impersonate="chrome")
    print(r.text)
```

**Supported Browsers:**

`curl_cffi` supports a wide range of browser versions for effective impersonation.  See the original repo for a full list.

**Bypass Cloudflare with API**

<a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>
[Yescaptcha](https://yescaptcha.com/i/stfnIO) is a proxy service that bypasses Cloudflare and uses the API interface to
obtain verified cookies (e.g. `cf_clearance`).

**Ecosystem & Integrations:**

*   **Scrapy:** `scrapy-curl-cffi`, `scrapy-impersonate`, `scrapy-fingerprint`
*   **Requests Adapters:** `curl-adapter`
*   **HTTPX Adapters:**  `httpx-curl-cffi`
*   **Captcha Resolvers:**  CapSolver, YesCaptcha

**Contribute**
Please use a different branch other than `main` and check the
"Allow edits by maintainers" box when creating pull requests.