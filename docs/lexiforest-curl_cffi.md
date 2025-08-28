# curl_cffi: The Fastest Python HTTP Client with Browser Impersonation

**Bypass bot detection and scrape the web with ease using `curl_cffi`, a high-performance Python binding for libcurl-impersonate.**  [Check out the original repo](https://github.com/lexiforest/curl_cffi).

[![PyPI Downloads](https://static.pepy.tech/badge/curl-cffi/week)](https://pepy.tech/projects/curl-cffi)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://img.shields.io/pypi/pyversions/curl_cffi)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi)
[![Generic badge](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Generic badge](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` provides a powerful and efficient way to make HTTP requests in Python, built on top of the `curl-impersonate` fork.  This library allows you to mimic browser behavior, making it ideal for web scraping and bypassing anti-bot measures.

**Key Features:**

*   **Browser Impersonation:** Emulates browser TLS/JA3 and HTTP/2 fingerprints, including Chrome, Safari, and others.
*   **High Performance:** Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`.
*   **Familiar API:** Uses a `requests`-like API, making it easy to learn and use.
*   **Asynchronous Support:** Includes `asyncio` support for non-blocking, concurrent requests, with proxy rotation.
*   **HTTP/2 & HTTP/3 Support:**  Offers support for the latest HTTP protocols.
*   **WebSocket Support:**  Enables real-time communication with WebSocket servers.
*   **Pre-compiled:**  No need to compile on your machine.
*   **MIT Licensed:**  Free to use and integrate into your projects.

**Feature Comparison**

| Feature            | requests | aiohttp | httpx  | pycurl     | curl_cffi |
| ------------------ | -------- | ------- | ------ | ---------- | --------- |
| http/2             | ❌       | ❌      | ✅     | ✅          | ✅        |
| http/3             | ❌       | ❌      | ❌     | ☑️<sup>1</sup> | ✅<sup>2</sup>       |
| sync               | ✅       | ❌      | ✅     | ✅          | ✅        |
| async              | ❌       | ✅      | ✅     | ❌          | ✅        |
| websocket          | ❌       | ✅      | ❌     | ❌          | ✅        |
| fingerprints       | ❌       | ❌      | ❌     | ❌          | ✅        |
| speed              | 🐇       | 🐇🐇     | 🐇     | 🐇🐇         | 🐇🐇       |

*Notes:*

1.  For pycurl, you need an HTTP/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2.  Since v0.11.4.

**Installation**

```bash
pip install curl_cffi --upgrade
```

**Usage Examples**

*   **Requests-like API:**

```python
import curl_cffi

r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())
```

*   **Sessions:**

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
```

*   **Asyncio:**

```python
from curl_cffi import AsyncSession
async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

*   **WebSockets:**

```python
from curl_cffi import WebSocket

def on_message(ws: WebSocket, message: str | bytes):
    print(message)

ws = WebSocket(on_message=on_message)
ws.run_forever("wss://api.gemini.com/v1/marketdata/BTCUSD")
```

**Supported Browser Versions**

`curl_cffi` supports the same browser versions as supported by my [fork](https://github.com/lexiforest/curl-impersonate) of [curl-impersonate](https://github.com/lwthiker/curl-impersonate):
If you see a version, e.g. `chrome135`, were skipped, you can simply impersonate it with your own headers and the previous version.

If you don't want to look up the headers etc, by yourself, consider buying commercial support from [impersonate.pro](https://impersonate.pro),
we have comprehensive browser fingerprints database for almost all the browser versions on various platforms.

If you are trying to impersonate a target other than a browser, use `ja3=...` and `akamai=...`
to specify your own customized fingerprints. See the [docs on impersonation](https://curl-cffi.readthedocs.io/en/latest/impersonate/_index.html) for details.

|Browser|Open Source| Pro version|
|---|---|---|
|Chrome|chrome99, chrome100, chrome101, chrome104, chrome107, chrome110, chrome116<sup>[1]</sup>, chrome119<sup>[1]</sup>, chrome120<sup>[1]</sup>, chrome123<sup>[3]</sup>, chrome124<sup>[3]</sup>, chrome131<sup>[4]</sup>, chrome133a<sup>[5][6]</sup>, chrome136<sup>[6]</sup>|chrome132, chrome134, chrome135|
|Chrome Android| chrome99_android, chrome131_android <sup>[4]</sup>|chrome132_android, chrome133_android, chrome134_android, chrome135_android|
|Chrome iOS|N/A|coming soon|
|Safari <sup>[7]</sup>|safari153 <sup>[2]</sup>, safari155 <sup>[2]</sup>, safari170 <sup>[1]</sup>, safari180 <sup>[4]</sup>, safari184 <sup>[6]</sup>, safari260 <sup>[8]</sup>|coming soon|
|Safari iOS <sup>[7]</sup>| safari172_ios<sup>[1]</sup>, safari180_ios<sup>[4]</sup>, safari184_ios <sup>[6]</sup>, safari260_ios <sup>[8]</sup>|coming soon|
|Firefox|firefox133<sup>[5]</sup>, firefox135<sup>[7]</sup>|coming soon|
|Firefox Android|N/A|firefox135_android|
|Tor|tor145 <sup>[7]</sup>|coming soon|
|Edge|edge99, edge101|edge133, edge135|
|Opera|N/A|coming soon|
|Brave|N/A|coming soon|


Notes:
1. Added in version `0.6.0`.
2. Fixed in version `0.6.0`, previous http2 fingerprints were [not correct](https://github.com/lwthiker/curl-impersonate/issues/215).
3. Added in version `0.7.0`.
4. Added in version `0.8.0`.
5. Added in version `0.9.0`.
6. The version postfix `-a`(e.g. `chrome133a`) means that this is an alternative version, i.e. the fingerprint has not been officially updated by browser, but has been observed because of A/B testing.
5. Added in version `0.10.0`.
6. Added in version `0.11.0`.
7. Since `0.11.0`, the format `safari184_ios` is preferred over `safari18_4_ios`, both are supported, but the latter is quite confusing and hard to parse.
8. Added in  `0.12.0`.

**Ecosystem Integrations**

*   **Scrapy:** `divtiply/scrapy-curl-cffi`, `jxlil/scrapy-impersonate`, `tieyongjie/scrapy-fingerprint`
*   **Adapters:** `el1s7/curl-adapter` (for requests), `vgavro/httpx-curl-cffi` (for httpx)
*   **Captcha Resolvers:** CapSolver, YesCaptcha

**Sponsors**

[Include the sponsor information here, using the original README's formatting.]

**Contributing**

[Include the contributing guidelines here, using the original README's formatting.]