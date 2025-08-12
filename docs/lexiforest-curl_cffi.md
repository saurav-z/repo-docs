# curl-cffi: The Fastest & Most Flexible Python HTTP Client with Browser Impersonation

**Bypass website restrictions and access the web like a real browser with `curl-cffi` - the powerful Python binding for `curl`!** [View the original repo](https://github.com/lexiforest/curl_cffi)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://curl-cffi.readthedocs.io)

`curl-cffi` provides a high-performance Python interface for interacting with web servers, built on top of the robust `curl` library. It offers browser impersonation capabilities, making it ideal for web scraping, bypassing bot detection, and testing.

**Key Features:**

*   🚀 **Blazing Fast:** Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl`.
*   🕵️ **Browser Impersonation:** Mimics browser fingerprints (TLS/JA3, HTTP/2) for effective bypassing of anti-bot measures. Supports Chrome, Safari, Firefox, and more.
*   💻 **Familiar API:**  Uses a `requests`-like API for ease of use.
*   📦 **Pre-compiled & Ready to Use:** No complex compilation steps required.
*   🔄 **Async Support:** Includes `asyncio` support with proxy rotation.
*   🌐 **Modern Protocol Support:**  Supports HTTP/2 and HTTP/3.
*   🕸️ **Websocket Support:** Includes both synchronous and asynchronous websocket support.
*   🛡️ **MIT Licensed:** Free to use and integrate.

**Compared to Other HTTP Clients:**

| Feature         | requests | aiohttp | httpx | pycurl | curl_cffi |
|-----------------|----------|---------|-------|--------|-----------|
| HTTP/2          | ❌       | ❌      | ✅    | ✅     | ✅        |
| HTTP/3          | ❌       | ❌      | ❌    | ☑️<sup>1</sup>   | ✅<sup>2</sup>     |
| Sync            | ✅       | ❌      | ✅    | ✅     | ✅        |
| Async           | ❌       | ✅      | ✅    | ❌     | ✅        |
| WebSocket       | ❌       | ✅      | ❌    | ❌     | ✅        |
| Fingerprints    | ❌       | ❌      | ❌    | ❌     | ✅        |
| Speed           | 🐇       | 🐇🐇     | 🐇    | 🐇🐇   | 🐇🐇       |

Notes:
1.  For pycurl, you need an http/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2.  Since v0.11.4.

**Installation:**

```bash
pip install curl_cffi --upgrade
```

**Basic Usage:**

```python
import curl_cffi

# Impersonate a Chrome browser
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Use a session to maintain cookies
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
r = s.get("https://httpbin.org/cookies")
print(r.json())
```

**Impersonation Examples:**

*   `impersonate="chrome"` (latest)
*   `impersonate="chrome124"` (specific version)
*   `impersonate="safari"`
*   `impersonate="realworld"` (random version based on market share - *pro feature*)
*   `ja3=...`, `akamai=...` (custom fingerprints)

**Asyncio Example:**

```python
from curl_cffi import AsyncSession
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com")
    print(r.text)
```

**WebSockets Example:**

```python
from curl_cffi import WebSocket

def on_message(ws: WebSocket, message: str | bytes):
    print(message)

ws = WebSocket(on_message=on_message)
ws.run_forever("wss://api.gemini.com/v1/marketdata/BTCUSD")
```

**Supported Browser Versions:**

`curl-cffi` supports a wide range of browser versions, as specified in the original [curl-impersonate](https://github.com/lwthiker/curl-impersonate) project. See the full list in the original README.

**Ecosystem Integrations:**
*   **Scrapy:**  [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapters:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi).
*   **Captcha Resolvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

**Commercial Support:**

For professional support and access to an extended database of browser fingerprints, visit [impersonate.pro](https://impersonate.pro).

**Contribute:**

Contributions are welcome! Please use a different branch than `main` and check "Allow edits by maintainers" for PRs.

---