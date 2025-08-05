# curl_cffi: The Fastest Python HTTP Client That Impersonates Browsers

**Bypass website restrictions with ease using curl_cffi, the Python library that lets you mimic browser fingerprints for secure and efficient web scraping and automation.**  [Check out the original repo!](https://github.com/lexiforest/curl_cffi)

## Key Features

*   **Browser Impersonation:** Emulates popular browsers (Chrome, Safari, Firefox, Edge, and more) to bypass bot detection and access websites.
*   **High Performance:** Significantly faster than `requests` and `httpx`, rivaling `aiohttp` and `pycurl`.
*   **Requests-like API:** Easy to learn and use, mirroring the familiar `requests` library.
*   **Asynchronous Support:** Built-in `asyncio` support for non-blocking operations and efficient concurrency, with proxy rotation on each request.
*   **HTTP/2 & HTTP/3 Support:** Handles the latest web protocols that `requests` doesn't.
*   **Websocket Support:** Includes synchronous and asynchronous websocket functionalities
*   **Cross-Platform:** Works on Linux, macOS, and Windows.
*   **MIT Licensed:** Open-source and freely usable.

## Why Use curl_cffi?

*   **Bypass Anti-Bot Systems:**  Impersonate browser fingerprints (TLS/JA3, HTTP/2) to avoid detection and access websites that block traditional Python HTTP clients.
*   **Speed and Efficiency:** Achieve faster web scraping and automation workflows.
*   **Modern Protocol Support:** Benefit from HTTP/2 and HTTP/3 support for enhanced performance and compatibility.
*   **Simple to Integrate:**  Offers a user-friendly API that closely resembles the popular `requests` library.

## Installation

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

## Usage

### Requests-like API

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Impersonate a specific Chrome version
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome124")

# Use proxies
proxies = {"https": "http://localhost:3128"}
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome", proxies=proxies)
```

### Sessions

```python
s = curl_cffi.Session()
s.get("https://httpbin.org/cookies/set/foo/bar")
print(s.cookies)
```

### Asyncio

```python
import asyncio
from curl_cffi import AsyncSession

async with AsyncSession() as s:
    r = await s.get("https://example.com")
```

## Supported Browsers

`curl_cffi` supports a wide range of browser versions. See the original repo for an updated list.  For more advanced browser fingerprints, you can also find commercial support at [impersonate.pro](https://impersonate.pro).

## Ecosystem and Integrations

*   **Scrapy:** Integrates with Scrapy using projects like `scrapy-curl-cffi`, `scrapy-impersonate`, and `scrapy-fingerprint`.
*   **Adapters:** Adapters for `requests` (`curl-adapter`) and `httpx` (`httpx-curl-cffi`).
*   **Captcha Solvers:** Compatible with captcha solving services like CapSolver and YesCaptcha.

## Sponsorship

This project is made possible by its contributors and sponsors. [Click here to sponsor](https://github.com/sponsors/lexiforest).