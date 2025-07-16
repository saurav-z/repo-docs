# curl_cffi: Python's Premier Library for Browser Impersonation with cffi

**Bypass restrictions and scrape the web with ease using `curl_cffi`, a Python library built for browser impersonation, offering unparalleled speed and flexibility.**  [Check out the original repo!](https://github.com/lexiforest/curl_cffi)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Generic badge](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Generic badge](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` is a Python binding for the `curl-impersonate` fork, built using `cffi`. It provides a powerful and efficient way to interact with web servers, especially when needing to mimic browser behavior.  Get commercial support at [impersonate.pro](https://impersonate.pro).

**Key Features:**

*   **Browser Impersonation:**  Mimics TLS/JA3 and HTTP/2 fingerprints of popular browsers (Chrome, Safari, Firefox, Edge, etc.) and custom fingerprints, allowing you to bypass anti-bot measures.
*   **Blazing Fast Performance:**  Significantly faster than `requests` and `httpx`, on par with `aiohttp` and `pycurl`, ideal for high-volume web scraping and data retrieval.
*   **Familiar API:**  Offers a `requests`-like API for easy integration and a smooth learning curve.
*   **Asyncio Support:**  Includes asynchronous support with proxy rotation for efficient concurrent requests.
*   **HTTP/2 & HTTP/3 and Websocket Support:**  Leverages the latest web technologies.
*   **No Compilation Required:** Pre-compiled for easy installation across Linux, macOS, and Windows.
*   **Open Source & MIT Licensed:**  Use, modify, and distribute freely.

**Why Choose `curl_cffi`?**

*   **Bypass Restrictions:** Overcome website blocks and captchas with browser impersonation.
*   **High Performance:** Achieve superior speed and efficiency in your web interactions.
*   **Easy to Use:** Get up and running quickly with a familiar API.
*   **Versatile:** Support for sync, async, and websocket.

**Install:**

```bash
pip install curl_cffi --upgrade
```

**Usage Examples:**

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Using Sessions
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
r = s.get("https://httpbin.org/cookies")
print(r.json())

# Asyncio example
from curl_cffi import AsyncSession
import asyncio
async with AsyncSession() as s:
    r = await s.get("https://example.com", impersonate="chrome")
    print(r.text)
```

**Supported Browsers:**

`curl_cffi` supports many browser versions, with more available via commercial support ([impersonate.pro](https://impersonate.pro)).

*   **Chrome:**  chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116, chrome119, chrome120, chrome123, chrome124, chrome131, chrome133a, chrome136
*   **Chrome Android:** chrome99_android, chrome131_android
*   **Safari:** safari153, safari155, safari170, safari180, safari184, safari260
*   **Safari iOS:** safari172_ios, safari180_ios, safari184_ios, safari260_ios
*   **Firefox:** firefox133, firefox135
*   **Edge:** edge99, edge101
*   **Tor:** tor145

**Bypass Cloudflare and Captchas:**

`curl_cffi` integrates seamlessly with various services:

*   **[SerpApi](https://serpapi.com/)**: Scrape search engines.
*   **[YesCaptcha](https://yescaptcha.com/i/stfnIO)**: Proxy service for bypassing Cloudflare.
*   **[CapSolver](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC)**: AI-powered captcha solver.

**Ecosystem:**

*   Integration with Scrapy
    *   [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi)
    *   [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate)
    *   [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint)
*   Adapters for requests and httpx
    *   [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter)
    *   [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi)
*   Integration with captcha resolvers:
    *   [CapSolver](https://docs.capsolver.com/en/api/)
    *   [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview)

**Acknowledgements:**

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
*   Headers/Cookies files from [httpx](https://github.com/encode/httpx/).
*   Asyncio support inspired by Tornado's curl http client.
*   Synchronous WebSocket API inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   Asynchronous WebSocket API inspired by [aiohttp](https://github.com/aio-libs/aiohttp).