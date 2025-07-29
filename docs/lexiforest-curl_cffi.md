# curl_cffi: Python Library for Advanced Web Scraping and Browser Impersonation

**Bypass website restrictions and scrape the web like a pro with `curl_cffi`, a high-performance Python library built for mimicking browser behavior.**  [Explore the original repository](https://github.com/lexiforest/curl_cffi).

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi.svg)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)
[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` is the premier Python binding for the `curl-impersonate` fork, providing advanced features for web scraping and bypassing anti-bot measures. It leverages `cffi` for optimal performance and offers robust browser impersonation capabilities.  For commercial support, visit [impersonate.pro](https://impersonate.pro).

**Key Features:**

*   **Browser Impersonation:**  Impersonate a wide range of browsers (Chrome, Safari, Firefox, Edge, and more) including specific versions, to bypass anti-bot defenses.  Easily adapt to the latest browser versions and market share with a simple `impersonate` parameter.
*   **High Performance:** Significantly faster than standard Python HTTP clients like `requests` and `httpx`, rivaling the speed of `aiohttp` and `pycurl`.  See our [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark) for details.
*   **Familiar API:** Mimics the `requests` API for easy integration and a minimal learning curve.
*   **Asynchronous Support:**  Built-in `asyncio` support with proxy rotation for efficient, concurrent scraping.
*   **HTTP/2 and HTTP/3:** Full support for modern HTTP protocols, offering improved performance and compatibility.
*   **WebSockets:**  Robust support for both synchronous and asynchronous WebSockets.
*   **Customizable Fingerprints:**  Use custom JA3/TLS and HTTP/2 fingerprints to precisely control your requests.
*   **Pre-compiled:** No need to compile on your machine, making installation straightforward.
*   **MIT License:** Use it freely in your projects.

**Comparison with Other Libraries**

| Feature          | requests | aiohttp | httpx | pycurl | curl_cffi |
|-------------------|----------|---------|-------|--------|-----------|
| HTTP/2           | ❌       | ❌       | ✅    | ✅     | ✅         |
| HTTP/3           | ❌       | ❌       | ❌    | ☑️<sup>1</sup>  | ✅<sup>2</sup>       |
| Sync             | ✅       | ❌       | ✅    | ✅     | ✅         |
| Async            | ❌       | ✅       | ✅    | ❌     | ✅         |
| WebSockets       | ❌       | ✅       | ❌    | ❌     | ✅         |
| Fingerprints     | ❌       | ❌       | ❌    | ❌     | ✅         |
| Speed            | 🐇      | 🐇🐇      | 🐇   | 🐇🐇   | 🐇🐇       |

Notes:
1. For pycurl, you need an http/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2. Since v0.11.4.

**Installation:**

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

For unstable versions:

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

On macOS, you may need to install these dependencies:

```bash
brew install zstd nghttp2
```

**Usage Examples:**

```python
import curl_cffi

# Basic GET request with Chrome impersonation
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Using sessions
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
```

```python
# Asyncio Example
import asyncio
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
    print(r.text)
```

**Supported Browsers**

`curl_cffi` supports a wide range of browser versions.  See the full list and versioning details in the [README on GitHub](https://github.com/lexiforest/curl_cffi#supported-impersonate-browsers).

**Ecosystem and Integrations**

*   **Scrapy Integration:** `divtiply/scrapy-curl-cffi`, `jxlil/scrapy-impersonate`, `tieyongjie/scrapy-fingerprint`
*   **Adapters:** `el1s7/curl-adapter` (requests), `vgavro/httpx-curl-cffi` (httpx)
*   **Captcha Solvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview)

**Sponsors**
Maintenance of this project is made possible by all the [contributors](https://github.com/lexiforest/curl_cffi/graphs/contributors) and [sponsors](https://github.com/sponsors/lexiforest).

**Promotional Offers:**

*   **SerpApi:** Scrape Google and other search engines with their fast and reliable API.
    <a href="https://serpapi.com/" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/serpapi.png" alt="SerpAPI" height="67" width="63"></a>
*   **Yescaptcha:** Bypass Cloudflare with their API interface.  Register [here](https://yescaptcha.com/i/stfnIO).
    <a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>

*   **CapSolver:** AI-powered Captcha solver for seamless scraping. Use the code **"CURL"** for a 6% balance bonus. Register [here](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC).
    <a href="https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/capsolver.jpg" alt="CapSolver" height="50" width="178"></a>

**Acknowledgement**

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi), which is under the MIT license.
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py), which is under the BSD license.
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).