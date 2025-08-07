# curl_cffi: The Ultimate Python Library for Mimicking Browsers and Bypassing Restrictions

**Tired of getting blocked?** `curl_cffi` allows you to effortlessly impersonate browsers, making your Python HTTP requests more robust and effective. Check out the original repo: [https://github.com/lexiforest/curl_cffi](https://github.com/lexiforest/curl_cffi)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/curl-cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/curl_cffi)](https://pypi.org/project/curl-cffi/)
[![PyPI version](https://badge.fury.io/py/curl-cffi.svg)](https://badge.fury.io/py/curl-cffi)
[![Telegram Group](https://img.shields.io/badge/Telegram%20Group-join-blue?logo=telegram)](https://t.me/+lL9n33eZp480MGM1)
[![Discord](https://img.shields.io/badge/Discord-join-purple?logo=blue)](https://discord.gg/kJqMHHgdn2)

[Documentation](https://curl-cffi.readthedocs.io) | [Commercial Support](https://impersonate.pro)

`curl_cffi` is a powerful Python binding for the `curl-impersonate` fork, providing unmatched control over HTTP requests, built using [cffi](https://cffi.readthedocs.io/en/latest/).

## Key Features

*   **Browser Impersonation:** Mimic various browser fingerprints (TLS/JA3, HTTP/2) for seamless web interaction.
*   **High Performance:**  Significantly faster than `requests` and `httpx`, comparable to `aiohttp` and `pycurl`.
*   **Familiar API:** Easy to use, with a `requests`-like API for quick adoption.
*   **Pre-compiled & Ready to Use:** No need to compile on your machine, simplifying installation.
*   **Asynchronous Support:** Includes `asyncio` support with proxy rotation for efficient, concurrent requests.
*   **HTTP/2 & HTTP/3 Support:**  Supports the latest HTTP protocols, which requests does not.
*   **WebSocket Support:**  Supports both synchronous and asynchronous WebSockets.
*   **Open Source & MIT Licensed:**  Free to use and modify.

## Comparison

| Feature         | requests | aiohttp | httpx | pycurl | curl_cffi |
|-----------------|----------|---------|-------|--------|-----------|
| HTTP/2          | ❌        | ❌       | ✅    | ✅     | ✅         |
| HTTP/3          | ❌        | ❌       | ❌    | ☑️<sup>1</sup> | ✅<sup>2</sup>      |
| Sync            | ✅        | ❌      | ✅    | ✅     | ✅        |
| Async           | ❌        | ✅      | ✅    | ❌     | ✅        |
| WebSocket       | ❌        | ✅      | ❌    | ❌     | ✅        |
| Fingerprints    | ❌        | ❌      | ❌    | ❌     | ✅        |
| Speed           | 🐇        | 🐇🐇     | 🐇   | 🐇🐇    | 🐇🐇       |

*Notes:*

1.  For pycurl, you need an http/3 enabled libcurl to make it work, while curl_cffi packages libcurl-impersonate inside Python wheels.
2.  Since v0.11.4.

## Installation

Install with pip:

```bash
pip install curl_cffi --upgrade
```

For beta releases:

```bash
pip install curl_cffi --upgrade --pre
```

For unstable versions (from GitHub):

```bash
git clone https://github.com/lexiforest/curl_cffi/
cd curl_cffi
make preprocess
pip install .
```

On macOS, you may need to install additional dependencies:

```bash
brew install zstd nghttp2
```

## Usage

### Requests-like API

```python
import curl_cffi

# Impersonate Chrome
r = curl_cffi.get("https://tls.browserleaks.com/json", impersonate="chrome")
print(r.json())

# Pin specific browser versions
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

### Supported Browsers

See the documentation for the full list of supported browsers and versions: [https://curl-cffi.readthedocs.io/en/latest/impersonate.html](https://curl-cffi.readthedocs.io/en/latest/impersonate.html)

## Ecosystem

*   **Scrapy Integration:** [divtiply/scrapy-curl-cffi](https://github.com/divtiply/scrapy-curl-cffi), [jxlil/scrapy-impersonate](https://github.com/jxlil/scrapy-impersonate) and [tieyongjie/scrapy-fingerprint](https://github.com/tieyongjie/scrapy-fingerprint).
*   **Adapters:** [el1s7/curl-adapter](https://github.com/el1s7/curl-adapter), [vgavro/httpx-curl-cffi](https://github.com/vgavro/httpx-curl-cffi)
*   **Captcha Solvers:** [CapSolver](https://docs.capsolver.com/en/api/), [YesCaptcha](https://yescaptcha.atlassian.net/wiki/spaces/YESCAPTCHA/overview).

## Contributing

Contributions are welcome! Please use a branch other than `main` and check the "Allow edits by maintainers" box for your pull requests.