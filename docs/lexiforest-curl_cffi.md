# curl_cffi: Effortlessly Impersonate Browsers and Supercharge Your Python HTTP Requests

Tired of getting blocked? **curl_cffi** is a high-performance Python library built on `curl-impersonate` that allows you to mimic browser fingerprints, enabling you to bypass bot detection and access websites more effectively.  [Check out the original repo](https://github.com/lexiforest/curl_cffi).

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)
[Documentation](https://curl-cffi.readthedocs.io)

**Key Features:**

*   **Browser Impersonation:**  Mimic TLS/JA3 and HTTP/2 fingerprints of various browsers (Chrome, Safari, Firefox, and more), making your requests appear authentic.
*   **High Performance:** Significantly faster than `requests` and `httpx`, offering performance comparable to `aiohttp` and `pycurl`. See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API for easy adoption and minimal learning curve.
*   **Pre-compiled:** Ready to use out-of-the-box, no need to compile on your machine.
*   **Asyncio Support:**  Seamlessly integrates with `asyncio` for asynchronous requests and proxy rotation.
*   **HTTP/2 & HTTP/3 Support:** Includes support for modern protocols, which `requests` does not.
*   **WebSocket Support:** Includes both synchronous and asynchronous WebSocket implementations.
*   **MIT License:**  Free to use and integrate into your projects.

**Comparison:**

| Feature           | requests | aiohttp | httpx | pycurl | curl_cffi |
| ----------------- | -------- | ------- | ----- | ------ | --------- |
| HTTP/2            | ❌       | ❌      | ✅    | ✅     | ✅        |
| HTTP/3            | ❌       | ❌      | ❌    | ☑️¹    | ✅²       |
| Sync              | ✅       | ❌      | ✅    | ✅     | ✅        |
| Async             | ❌       | ✅      | ✅    | ❌     | ✅        |
| WebSocket         | ❌       | ✅      | ❌    | ❌     | ✅        |
| Fingerprints      | ❌       | ❌      | ❌    | ❌     | ✅        |
| Speed             | 🐇       | 🐇🐇     | 🐇    | 🐇🐇    | 🐇🐇       |

*Notes:
1. For pycurl, you need an http/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2. Since v0.11.4.*

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

```python
# Asyncio
from curl_cffi import AsyncSession
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com", impersonate="chrome")
    print(r.status_code)
```

**Supported Browsers:**

`curl_cffi` supports the same browser versions as supported by [curl-impersonate](https://github.com/lwthiker/curl-impersonate) and includes:

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

*Notes:
1. Added in version `0.6.0`.
2. Fixed in version `0.6.0`, previous http2 fingerprints were [not correct](https://github.com/lwthiker/curl-impersonate/issues/215).
3. Added in version `0.7.0`.
4. Added in version `0.8.0`.
5. Added in version `0.9.0`.
6. The version postfix `-a`(e.g. `chrome133a`) means that this is an alternative version, i.e. the fingerprint has not been officially updated by browser, but has been observed because of A/B testing.
5. Added in version `0.10.0`.
6. Added in version `0.11.0`.
7. Since `0.11.0`, the format `safari184_ios` is preferred over `safari18_4_ios`, both are supported, but the latter is quite confusing and hard to parse.
8. Added in  `0.12.0`.*

**Commercial Support:**

For comprehensive browser fingerprint databases and advanced features, consider commercial support at [impersonate.pro](https://impersonate.pro).

**Ecosystem:**

*   Integration with Scrapy: [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   Integrating with [requests](https://github.com/el1s7/curl-adapter), [httpx](https://github.com/vgavro/httpx-curl-cffi) as adapter.
*   Integrating with captcha resolvers: [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).