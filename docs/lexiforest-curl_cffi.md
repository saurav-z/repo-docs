# curl_cffi: The Fastest & Most Advanced Python HTTP Client with Browser Impersonation

Tired of getting blocked while web scraping? **`curl_cffi` allows you to seamlessly impersonate browsers and bypass anti-bot measures.**  This powerful Python library provides a robust and high-performance alternative to `requests` and `httpx`, allowing you to make HTTP requests that mimic browser behavior.  Check out the [original repo](https://github.com/lexiforest/curl_cffi) for the source code.

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://pypi.org/project/curl-cffi/)
[![Generic badge](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Generic badge](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io)

## Key Features

*   **Browser Impersonation:**  Mimic various browsers' TLS/JA3 and HTTP/2 fingerprints, including Chrome, Safari, Firefox, and more, to avoid detection.
*   **Blazing Fast Performance:** Significantly faster than `requests` and `httpx`, offering performance comparable to `aiohttp` and `pycurl`.
*   **Familiar API:** Uses a `requests`-like API for ease of use and a gentle learning curve.
*   **Asyncio Support:**  Built-in support for asynchronous operations with proxy rotation.
*   **HTTP/2 & HTTP/3 Support:**  Leverages the latest HTTP protocols for improved speed and efficiency.
*   **WebSockets:**  Supports WebSockets for real-time data streaming.
*   **Pre-compiled:** No need to compile on your machine.

## Why Choose curl_cffi?

| Feature         | requests | aiohttp | httpx | pycurl | curl_cffi |
|-----------------|----------|---------|-------|--------|-----------|
| HTTP/2          | ❌       | ❌       | ✅    | ✅     | ✅        |
| HTTP/3          | ❌       | ❌       | ❌    | ☑️<sup>1</sup>     | ✅<sup>2</sup>     |
| Sync            | ✅       | ❌       | ✅    | ✅     | ✅        |
| Async           | ❌       | ✅       | ✅    | ❌     | ✅        |
| WebSocket       | ❌       | ✅       | ❌    | ❌     | ✅        |
| Fingerprints    | ❌       | ❌       | ❌    | ❌     | ✅        |
| Speed           | 🐇       | 🐇🐇      | 🐇    | 🐇🐇   | 🐇🐇      |

**Notes:**
1.  For pycurl, you need an http/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2.  Since v0.11.4.

## Installation

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

For the unstable version from GitHub:

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

## Usage

### Requests-like API

```python
import curl_cffi

# Notice the impersonate parameter
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# To keep using the latest browser version as `curl_cffi` updates,
# simply set impersonate="chrome" without specifying a version.
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")

# To pin a specific version, use version numbers together.
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# http/socks proxies are supported
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()

# httpbin is a http test website, this endpoint makes the server set cookies
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
# <Cookies[<Cookie foo=bar for httpbin.org />]>

# retrieve cookies again to verify
r = s.get("https://httpbin.org/cookies")
print(r.json())
# {'cookies': {'foo': 'bar'}}
```

### Supported Browsers

`curl_cffi` supports the same browser versions as supported by [curl-impersonate](https://github.com/lwthiker/curl-impersonate):

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

### Asyncio Example

```python
from curl_cffi import AsyncSession
import asyncio

async with AsyncSession() as s:
    r = await s.get("https://example.com")
    print(r.status_code)

async with AsyncSession() as s:
    ws = await s.ws_connect("wss://echo.websocket.org")
    await asyncio.gather(*[ws.send_str("Hello, World!") for _ in range(10)])
    async for message in ws:
        print(message)
```

## Ecosystem Integrations

*   **Scrapy:**  [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate), [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapters:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter) for requests, [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi) for httpx.
*   **Captcha Resolvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Acknowledgements

-   Originally forked from [multippt/python_curl_cffi](https://github.com/multippt/python_curl_cffi).
-   Headers/Cookies files are copied from [httpx](https://github.com/encode/httpx/blob/master/httpx/_models.py).
-   Asyncio support is inspired by Tornado's curl http client.
-   The synchronous WebSocket API is inspired by [websocket_client](https://github.com/websocket-client/websocket-client).
-   The asynchronous WebSocket API is inspired by [aiohttp](https://github.com/aio-libs/aiohttp).

```

Key improvements and SEO considerations:

*   **Clear & Concise Title:**  The title clearly states the primary benefit.
*   **SEO-Friendly Description:**  The description includes relevant keywords ("Python HTTP client," "browser impersonation," "web scraping," etc.).
*   **Targeted Keywords:** Keywords are naturally integrated throughout the text.
*   **Feature Highlighting:**  Key features are presented in a bulleted list for easy readability and scannability.
*   **Benefit-Driven Language:** The text focuses on the *benefits* of using `curl_cffi` (e.g., bypassing blocks, faster performance).
*   **Clear Headings:**  Organized with clear headings and subheadings for better structure.
*   **Concise & Actionable Code Examples:**  Examples are short and easy to understand.
*   **Links:**  Includes the main repository link and other relevant links for reference.
*   **Mobile-Friendly Formatting:**  Uses markdown for better readability across devices.
*   **Contextual Table:** The table comparing `curl_cffi` to other libraries adds value.
*   **Strong Call to Action:** The opening sentence works as a hook, emphasizing the value proposition.
*   **Eliminated Promotional Content:** Removed the specific sponsor and partner promotions to ensure a more general and useful README. The YesCaptcha, SerpAPI and CapSolver sections have been removed as they were promotional and not core to the functionality.

This improved README is much more effective at attracting users and explaining the value of `curl_cffi`.