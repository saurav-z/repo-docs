# curl_cffi: Python's Fastest and Most Versatile HTTP Client for Web Scraping and Browser Impersonation

**Tired of getting blocked?** `curl_cffi` allows you to bypass website restrictions by impersonating various browsers' fingerprints, providing unparalleled speed and flexibility for your web interactions. [See the original repo](https://github.com/lexiforest/curl_cffi).

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi)
[![Generic badge](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Generic badge](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)
[Documentation](https://curl-cffi.readthedocs.io)

Built upon the powerful [curl-impersonate fork](https://github.com/lexiforest/curl-impersonate) and utilizing [cffi](https://cffi.readthedocs.io/en/latest/), `curl_cffi` provides a robust and efficient way to interact with websites. For commercial support and advanced features, explore [impersonate.pro](https://impersonate.pro).

**Key Features:**

*   **Browser Impersonation:** Mimics TLS/JA3 and HTTP/2 fingerprints of popular browsers (Chrome, Safari, Firefox, etc.) to avoid detection and bypass anti-scraping measures.
*   **Blazing Fast:** Outperforms `requests` and `httpx`, rivaling `aiohttp` and `pycurl` in speed.  See our [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Offers a `requests`-like API, making it easy to learn and integrate.
*   **Pre-compiled:** No need to compile on your machine; ready to use out-of-the-box.
*   **Asynchronous Support:** Supports `asyncio` for non-blocking operations and proxy rotation.
*   **HTTP/2 & HTTP/3 & WebSockets:** Includes modern protocol support for enhanced compatibility and performance.
*   **Websocket support:** Can use websockets.
*   **Open Source:** MIT licensed.

**Comparison with Other HTTP Clients:**

| Feature         | requests | aiohttp | httpx | pycurl | curl_cffi |
| --------------- | -------- | ------- | ----- | ------ | --------- |
| HTTP/2          | ❌       | ❌      | ✅    | ✅     | ✅        |
| HTTP/3          | ❌       | ❌      | ❌    | ☑️<sup>1</sup> | ✅<sup>2</sup>       |
| Sync            | ✅       | ❌      | ✅    | ✅     | ✅        |
| Async           | ❌       | ✅      | ✅    | ❌     | ✅        |
| WebSockets      | ❌       | ✅      | ❌    | ❌     | ✅        |
| Fingerprints    | ❌       | ❌      | ❌    | ❌     | ✅        |
| Speed           | 🐇       | 🐇🐇     | 🐇    | 🐇🐇    | 🐇🐇       |

**Notes:**
1.  For pycurl, you need an http/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2.  Since v0.11.4.

**Installation:**

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

For unstable releases:

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

On macOS, install dependencies:

```bash
brew install zstd nghttp2
```

**Usage:**

`curl_cffi` provides both low-level `curl` and high-level `requests`-like APIs.  See the original README for detailed examples, including examples for browser impersonation, session handling, asyncio, and websockets.

**Supported Impersonated Browsers:**

`curl_cffi` supports browser versions compatible with the [curl-impersonate](https://github.com/lwthiker/curl-impersonate) project. Check the original repo for an up-to-date browser version table (same table included in original README).

**Ecosystem & Integrations:**

*   Scrapy:  [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint)
*   Adapters for: [requests](https://github.com/el1s7/curl-adapter), [httpx](https://github.com/vgavro/httpx-curl-cffi)
*   Captcha Resolvers: [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview)

**Acknowledgements:**

(As per the original README)