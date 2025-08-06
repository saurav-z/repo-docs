# curl_cffi: Powerful Python Library for Browser Impersonation with cURL

Tired of getting blocked? **curl_cffi** is a blazing-fast Python library built on `curl-impersonate` that lets you mimic browser fingerprints, bypass anti-bot measures, and access web resources like a real user.  [Explore the original repo](https://github.com/lexiforest/curl_cffi).

## Key Features:

*   **Browser Impersonation:**  Mimics popular browsers (Chrome, Safari, Firefox, Edge, etc.) to bypass bot detection, including TLS/JA3 and HTTP/2 fingerprints.
*   **Blazing Fast Performance:** Outperforms `requests`, `httpx`, and comparable to `aiohttp` and `pycurl`, as shown in [benchmarks](https://github.com/lexiforest/curl_cffi/tree/main/benchmark).
*   **Familiar API:**  Uses a `requests`-like API, making it easy to integrate into your existing projects.
*   **Asyncio Support:**  Offers full `asyncio` support with proxy rotation for asynchronous web requests.
*   **HTTP/2 & HTTP/3 Support:**  Supports modern HTTP protocols for improved performance.
*   **WebSockets:** Includes robust support for both synchronous and asynchronous WebSockets.
*   **Pre-compiled & Easy Installation:**  Ready to use out-of-the-box with pre-compiled binaries, simplifying setup.

## Benefits:

*   **Bypass Anti-Bot Systems:**  Effectively bypasses Cloudflare and other anti-bot systems.
*   **Enhanced Scraping:**  Allows for reliable web scraping of websites that employ bot detection.
*   **Web Application Testing:** Enables testing and validation of web application behavior under various user scenarios.
*   **Competitive Edge:** Offers an advantage in accessing protected web content and services.

## Installation

Install with pip:

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

## Usage Examples:

**requests-like API (Synchronous):**

```python
import curl_cffi

r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())
```

**Asyncio API (Asynchronous):**

```python
import asyncio
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
    print(r.text)
```

##  Bypass Cloudflare with API

<a href="https://yescaptcha.com/i/stfnIO" target="_blank"><img src="https://raw.githubusercontent.com/lexiforest/curl_cffi/main/assets/yescaptcha.png" alt="Yes Captcha!" height="47" width="149"></a>

Yescaptcha is a proxy service that bypasses Cloudflare and uses the API interface to
obtain verified cookies (e.g. `cf_clearance`). Click [here](https://yescaptcha.com/i/stfnIO)
to register: https://yescaptcha.com/i/stfnIO

## Supported Impersonation Profiles

`curl_cffi` supports impersonating the following browsers:
*   Chrome
*   Safari
*   Firefox
*   Edge
*   Opera
*   Brave

For specific version support, please refer to the [documentation](https://curl-cffi.readthedocs.io/en/latest/impersonate.html) or the table in the original README.

## Ecosystem & Integrations:

*   **Scrapy Integration:** `scrapy-curl-cffi`, `scrapy-impersonate`, `scrapy-fingerprint`.
*   **Adapter for Existing Libraries:**  `curl-adapter` (for `requests`), `httpx-curl-cffi` (for `httpx`).
*   **Captcha Resolvers:**  Integrates with services such as CapSolver and YesCaptcha.

## Support & Resources:

*   [Documentation](https://curl-cffi.readthedocs.io)
*   [Telegram Group](https://t.me/+lL9n33eZp480MGM1)
*   [Discord](https://discord.gg/kJqMHHgdn2)

Get started today and unlock the power of browser impersonation with `curl_cffi`!