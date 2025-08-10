# curl-cffi: The Ultimate Python Library for Web Scraping and API Interaction

**Bypass website blocks and effortlessly impersonate browsers with curl-cffi, a high-performance Python binding for curl.**  [View the original repository](https://github.com/lexiforest/curl_cffi)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io) | [Commercial Support](https://impersonate.pro)

`curl-cffi` is the leading Python library for web scraping and API interaction, built on top of the powerful `curl` and `curl-impersonate` forks. It provides an incredibly flexible and efficient way to interact with websites, mimicking real browser behavior to overcome anti-scraping measures.

## Key Features:

*   **Browser Impersonation:** Emulates popular browsers (Chrome, Safari, Firefox, Edge, Tor) with accurate JA3/TLS and HTTP/2 fingerprints.  Easily bypass bot detection.
*   **High Performance:** Outperforms `requests` and `httpx`, offering speed comparable to `aiohttp` and `pycurl`. See benchmarks [here](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a user-friendly, `requests`-like API, reducing the learning curve.
*   **Asyncio Support:** Enables asynchronous operations with proxy rotation for maximum efficiency.
*   **HTTP/2 & HTTP/3 Support:** Includes support for the latest protocols, providing superior speed and compatibility.
*   **WebSocket Support:** Provides both synchronous and asynchronous WebSocket capabilities.
*   **Pre-compiled:** Includes pre-compiled binaries for easy installation across various platforms.

## Installation

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

## Core Functionality & Examples

### `requests`-like API

```python
import curl_cffi

r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Impersonate specific browser versions
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Proxy support
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)

r = s.get("https://httpbin.org/cookies")
print(r.json())
```

### Asyncio Support

```python
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

## Supported Browser Fingerprints
  
`curl_cffi` seamlessly integrates the fingerprints supported by its [fork](https://github.com/lexiforest/curl-impersonate) of [curl-impersonate](https://github.com/lwthiker/curl-impersonate):

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

*   Note: Commercial support is available for an extensive database of browser fingerprints.  Visit [impersonate.pro](https://impersonate.pro) for details.

## Ecosystem Integrations

*   **Scrapy:**  [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapters for requests and httpx:**  [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi).
*   **Captcha Resolvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Sponsors and Contributors

This project is made possible by the dedication of its [contributors](https://github.com/lexiforest/curl_cffi/graphs/contributors) and [sponsors](https://github.com/sponsors/lexiforest).

## Contributing

Contributions are welcome!  Please create a branch other than `main` and check "Allow edits by maintainers" in your pull request.

---