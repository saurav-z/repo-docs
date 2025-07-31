# curl_cffi: Python's Premier Library for Browser-Like Web Scraping and Impersonation

**Bypass website restrictions and scrape the web with ease using `curl_cffi`, a powerful Python library that mimics browser fingerprints.**

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi)
[![Generic badge](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Generic badge](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io) | [Original Repo](https://github.com/lexiforest/curl_cffi)

`curl_cffi` is a Python binding for the highly effective `curl-impersonate` fork, built via `cffi`, empowering you to overcome website defenses that often block standard HTTP clients. If you encounter blocks while scraping, `curl_cffi` is your go-to solution!

**Key Features:**

*   **Browser Impersonation:** Mimics TLS/JA3 and HTTP/2 fingerprints of popular browsers (Chrome, Safari, Firefox, etc.) and supports custom fingerprints.
*   **High Performance:** Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`.  See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API for easy adoption.
*   **Pre-compiled:** No need to compile on your machine; it's ready to go.
*   **Asynchronous Support:** Offers full `asyncio` support with proxy rotation.
*   **HTTP/2 & HTTP/3 Support:**  Supports both HTTP/2 and HTTP/3, unlike some other libraries.
*   **Websocket Support:** Provides robust websocket functionality.
*   **MIT License:**  Free to use and open-source.

**Bypass common restrictions**

*   Bypass Cloudflare
*   Solve Captchas 

## Installation

```bash
pip install curl_cffi --upgrade
```

**Important:** For commercial support, visit [impersonate.pro](https://impersonate.pro).

## Usage Examples

```python
import curl_cffi

# Impersonate Chrome (latest version)
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Specify a specific browser version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome120")

# Use sessions for persistent cookies
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
r = s.get("https://httpbin.org/cookies")
print(r.json())

# Asyncio example
import asyncio
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com", impersonate="chrome")
    print(r.text)
```

## Ecosystem & Integrations

*   **Scrapy Integration:**  Seamlessly integrate `curl_cffi` with Scrapy using libraries like [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi).
*   **Adapters:**  Use with `requests` and `httpx` via adapters:  [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi)
*   **Captcha Resolvers:** Integrate with captcha solvers: [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Sponsors

Maintenance of this project is made possible by all the <a href="https://github.com/lexiforest/curl_cffi/graphs/contributors">contributors</a> and <a href="https://github.com/sponsors/lexiforest">sponsors</a>. If you'd like to sponsor this project and have your avatar or company logo appear below <a href="https://github.com/sponsors/lexiforest">click here</a>. 💖

------

<a href="https://serpapi.com/" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/serpapi.png" alt="SerpAPI" height="67" width="63"></a>

Scrape Google and other search engines from [SerpApi](https://serpapi.com/)'s fast, easy, and complete API. 0.66s average response time (≤ 0.5s for Ludicrous Speed Max accounts), 99.95% SLAs, pay for successful responses only.

------

### Bypass Cloudflare with API

<a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>

Yescaptcha is a proxy service that bypasses Cloudflare and uses the API interface to
obtain verified cookies (e.g. `cf_clearance`). Click [here](https://yescaptcha.com/i/stfnIO)
to register: https://yescaptcha.com/i/stfnIO

------

### Easy Captcha Bypass for Scraping

<a href="https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/capsolver.jpg" alt="CapSolver" height="50" width="178"></a>

[CapSolver](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC)
is an AI-powered tool that easily bypasses Captchas, allowing uninterrupted access to
public data. It supports a variety of Captchas and works seamlessly with `curl_cffi`,
Puppeteer, Playwright, and more. Fast, reliable, and cost-effective. Plus, `curl_cffi`
users can use the code **"CURL"** to get an extra 6% balance! and register
[here](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC).

------

## Supported Browsers

`curl_cffi` supports the same browser versions as supported by my [fork](https://github.com/lexiforest/curl-impersonate) of [curl-impersonate](https://github.com/lwthiker/curl-impersonate):

Open source version of curl_cffi includes versions whose fingerprints differ from previous versions.
If you see a version, e.g. `chrome135`, were skipped, you can simply impersonate it with your own headers and the previous version.

If you don't want to look up the headers etc, by yourself, consider buying commercial support from [impersonate.pro](https://impersonate.pro),
we have comprehensive browser fingerprints database for almost all the browser versions on various platforms.

If you are trying to impersonate a target other than a browser, use `ja3=...` and `akamai=...`
to specify your own customized fingerprints. See the [docs on impersonation](https://curl-cffi.readthedocs.io/en/latest/impersonate.html) for details.

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

## Acknowledgements

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi), which is under the MIT license.
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py), which is under the BSD license.
*   Asyncio support is inspired by Tornado's curl http client.
*   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
*   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).