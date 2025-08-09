# curl_cffi: The Fastest Python HTTP Client with Browser Impersonation

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

**Bypass website restrictions and achieve lightning-fast web requests with `curl_cffi`, a Python library leveraging the power of `curl-impersonate`.**

[Documentation](https://curl-cffi.readthedocs.io) | [GitHub Repository](https://github.com/lexiforest/curl_cffi)

`curl_cffi` provides a high-performance Python binding for the `curl-impersonate` fork, using `cffi`.  It's designed for speed and flexibility, enabling you to mimic browser behavior and overcome anti-bot measures.  For commercial support, visit [impersonate.pro](https://impersonate.pro).

**Key Features:**

*   **Browser Impersonation:** Mimic various browsers' TLS/JA3 and HTTP/2 fingerprints to bypass bot detection.
*   **Blazing Fast:** Outperforms `requests` and `httpx`, comparable to `aiohttp` and `pycurl`. See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API for ease of use.
*   **Pre-compiled:**  No need to compile on your machine.
*   **Asyncio Support:**  Offers asynchronous requests with proxy rotation.
*   **HTTP/2 & HTTP/3 Support:** Includes support for modern protocols.
*   **Websocket Support:** Provides websocket client functionality.
*   **Open Source:**  MIT licensed, allowing for free use and modification.

**[YesCaptcha](https://yescaptcha.com/i/stfnIO) - Bypass Cloudflare with API:**

[<img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149">](https://yescaptcha.com/i/stfnIO)

Yescaptcha is a proxy service that bypasses Cloudflare and uses the API interface to obtain verified cookies (e.g. `cf_clearance`). Click [here](https://yescaptcha.com/i/stfnIO) to register.

**Installation:**

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

**Usage Examples:**

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Use sessions
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
```

**Supported Impersonation Browsers:**

`curl_cffi` supports a variety of browser versions, constantly updated to reflect the latest browser fingerprints.  See the original README for details.

**Ecosystem Integrations:**

*   Scrapy: [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   Requests Adapter: [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter)
*   Httpx Adapter: [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi)
*   Captcha Resolvers: CapSolver, YesCaptcha

**Contribute:**

Contributions are welcome! Please submit your PRs on a separate branch, and check the "Allow edits by maintainers" box.