# curl_cffi: Python Library for Mimicking Browser Behavior with `curl`

**Bypass website restrictions and enhance your web scraping capabilities with `curl_cffi`, the powerful Python binding for `curl` that lets you impersonate browsers' fingerprints.** ([Original Repo](https://github.com/lexiforest/curl_cffi))

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi)
[![Generic badge](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Generic badge](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

`curl_cffi` offers a robust Python interface to the `curl-impersonate` fork, providing advanced control over HTTP requests, making it ideal for web scraping, bypassing bot detection, and testing web applications.

**Key Features:**

*   **Browser Impersonation:** Mimic various browser fingerprints (TLS/JA3, HTTP/2) including Chrome, Safari, Firefox and more to avoid blocking.
*   **High Performance:** Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl`. See [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:** Uses a `requests`-like API, making it easy to learn and use.
*   **Asyncio Support:** Offers `asyncio` integration with proxy rotation for concurrent requests.
*   **HTTP/2 & HTTP/3 & WebSockets:** Supports modern protocols, including HTTP/2, HTTP/3 and WebSockets, which other libraries might lack.
*   **Pre-compiled:** No need to compile on your machine.

## Installation

```bash
pip install curl_cffi --upgrade
```

## Usage Examples

### Requests-like API

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Use Sessions to maintain cookies
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
r = s.get("https://httpbin.org/cookies")
print(r.json())

# Use proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Asyncio

```python
import asyncio
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

### WebSockets

```python
from curl_cffi import WebSocket

def on_message(ws: WebSocket, message: str | bytes):
    print(message)

ws = WebSocket(on_message=on_message)
ws.run_forever("wss://api.gemini.com/v1/marketdata/BTCUSD")
```

## Ecosystem Integrations

*   **Scrapy:** `scrapy-curl-cffi`, `scrapy-impersonate`, `scrapy-fingerprint`
*   **Adapters:** `curl-adapter` (for `requests`), `httpx-curl-cffi` (for `httpx`)
*   **Captcha Solvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview)

## Commercial Support

For commercial support and comprehensive browser fingerprint databases, visit [impersonate.pro](https://impersonate.pro).

## Sponsors

Maintenance of this project is made possible by all the [contributors](https://github.com/lexiforest/curl_cffi/graphs/contributors) and [sponsors](https://github.com/sponsors/lexiforest). If you'd like to sponsor this project and have your avatar or company logo appear below [click here](https://github.com/sponsors/lexiforest). 💖

## Integrations & Services

Leverage these services to enhance your scraping and bypass capabilities:

### Scrape Google with SerpApi

<a href="https://serpapi.com/" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/serpapi.png" alt="SerpAPI" height="67" width="63"></a>

Scrape Google and other search engines from [SerpApi](https://serpapi.com/)'s fast, easy, and complete API.

------

### Bypass Cloudflare with Yescaptcha API

<a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>

Yescaptcha is a proxy service that bypasses Cloudflare and uses the API interface to
obtain verified cookies (e.g. `cf_clearance`). Click [here](https://yescaptcha.com/i/stfnIO)
to register: https://yescaptcha.com/i/stfnIO

------

### Easy Captcha Bypass for Scraping with CapSolver

<a href="https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/capsolver.jpg" alt="CapSolver" height="50" width="178"></a>

[CapSolver](https://dashboard.capsolver.com/passport/register?inviteCode=0FLEay4iroNC)
is an AI-powered tool that easily bypasses Captchas, allowing uninterrupted access to
public data. Use code **"CURL"** for an extra 6% balance!

## Supported Browser Fingerprints

Supports same browser versions as supported by [curl-impersonate](https://github.com/lwthiker/curl-impersonate)

See the [docs on impersonation](https://curl-cffi.readthedocs.io/en/latest/impersonate.html) for details.

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

## Acknowledgements

*   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi)
*   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py)
*   Asyncio support inspired by Tornado and aiohttp.
*   Sync WebSockets inspired by [websocket_client](https://github.com/websocket-client/websocket-client).

---