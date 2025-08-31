<p align="center">
    <img src="https://github.com/user-attachments/assets/219f2dbc-37ed-4aea-a289-ba39cdbb335d" alt="Pydoll Logo" /> <br>
</p>
<h1 align="center">Pydoll: Automate the Web, Naturally</h1>

<p align="center">
    <a href="https://github.com/autoscrape-labs/pydoll/stargazers"><img src="https://img.shields.io/github/stars/autoscrape-labs/pydoll?style=social"></a>
    <a href="https://codecov.io/gh/autoscrape-labs/pydoll" >
        <img src="https://codecov.io/gh/autoscrape-labs/pydoll/graph/badge.svg?token=40I938OGM9"/>
    </a>
    <img src="https://github.com/autoscrape-labs/pydoll/actions/workflows/tests.yml/badge.svg" alt="Tests">
    <img src="https://github.com/autoscrape-labs/pydoll/actions/workflows/ruff-ci.yml/badge.svg" alt="Ruff CI">
    <img src="https://github.com/autoscrape-labs/pydoll/actions/workflows/mypy.yml/badge.svg" alt="MyPy CI">
    <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="Python >= 3.10">
    <a href="https://deepwiki.com/autoscrape-labs/pydoll"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>


<p align="center">
  📖 <a href="https://pydoll.tech/">Documentation</a> •
  🚀 <a href="#-getting-started">Getting Started</a> •
  ⚡ <a href="#-advanced-features">Advanced Features</a> •
  🤝 <a href="#-contributing">Contributing</a> •
  💖 <a href="#-support-my-work">Support My Work</a>
</p>

**Pydoll simplifies web automation by connecting directly to the Chrome DevTools Protocol, allowing you to interact with web pages naturally and efficiently.**

## Key Features

*   **Zero WebDriver Dependency:** Eliminate compatibility headaches with direct CDP integration.
*   **Human-Like Interaction Engine:** Navigate and interact with websites as a real user would, even bypassing behavioral CAPTCHAs.
*   **Asynchronous Performance:** Automate at high speeds, handling multiple tasks concurrently.
*   **Intuitive API:** Focus on your automation logic, not complex configurations.
*   **Remote Browser Control:** Connect to and control Chrome instances running remotely.

[Back to Top](#pydoll-automate-the-web-naturally)

## 🚀 Getting Started

Get up and running quickly with Pydoll. Install with pip:

```bash
pip install pydoll-python
```

See the full [installation instructions](https://pydoll.tech/docs/getting-started/#installation) in the documentation.

### Example: Google Search & Result Click

This example demonstrates how to perform a Google search and click the first result:

```python
import asyncio

from pydoll.browser import Chrome
from pydoll.constants import Key

async def google_search(query: str):
    async with Chrome() as browser:
        tab = await browser.start()
        await tab.go_to('https://www.google.com')
        search_box = await tab.find(tag_name='textarea', name='q')
        await search_box.insert_text(query)
        await search_box.press_keyboard_key(Key.ENTER)
        await (await tab.find(
            tag_name='h3',
            text='autoscrape-labs/pydoll',
            timeout=10,
        )).click()
        await tab.find(id='repository-container-header', timeout=10)

asyncio.run(google_search('pydoll python'))
```

## ⚡ Advanced Features

Pydoll offers advanced features for complete automation control.

### Remote Connections via WebSocket

Control any Chrome instance remotely using its WebSocket address:

```python
from pydoll.browser.chromium import Chrome

chrome = Chrome()
tab = await chrome.connect('ws://YOUR_HOST:9222/devtools/browser/XXXX')

# Full power unlocked: navigation, element automation, requests, events…
await tab.go_to('https://example.com')
title = await tab.execute_script('return document.title')
print(title)
```

### Navigate the DOM like a pro: get_children_elements() and get_siblings_elements()

Two delightful helpers to traverse complex layouts with intention:

```python
# Grab direct children of a container
container = await tab.find(id='cards')
cards = await container.get_children_elements(max_depth=1)

# Want to go deeper? This will return children of children (and so on)
elements = await container.get_children_elements(max_depth=2) 

# Walk horizontal lists without re-querying the DOM
active = await tab.find(class_name='item-active')
siblings = await active.get_siblings_elements()

print(len(cards), len(siblings))
```

### WebElement: state waiting and new public APIs

- New `wait_until(...)` on `WebElement` to await element states with minimal code:

```python
# Wait until it becomes visible OR the timeout expires
await element.wait_until(is_visible=True, timeout=5)

# Wait until it becomes interactable (visible, on top, receiving pointer events)
await element.wait_until(is_interactable=True, timeout=10)
```

- Methods now public on `WebElement`:
  - `is_visible()`
    - Checks that the element has a visible area (> 0), isn’t hidden by CSS and is in the viewport (after `scroll_into_view()` when needed). Useful pre-check before interactions.
  - `is_interactable()`
    - “Click-ready” state: combines visibility, enabledness and pointer-event hit testing. Ideal for robust flows that avoid lost clicks.
  - `is_on_top()`
    - Verifies the element is the top hit-test target at the intended click point, avoiding overlays.
  - `execute_script(script: str, return_by_value: bool = False)`
    - Executes JavaScript in the element’s own context (where `this` is the element). Great for fine-tuning and quick inspections.

```python
# Visually outline the element via JS
await element.execute_script("this.style.outline='2px solid #22d3ee'")

# Confirm states
visible = await element.is_visible()
interactable = await element.is_interactable()
on_top = await element.is_on_top()
```

### Advanced Element Search

```python
import asyncio
from pydoll.browser import Chrome

async def element_finding_examples():
    async with Chrome() as browser:
        tab = await browser.start()
        await tab.go_to('https://example.com')

        # Find by attributes (most intuitive)
        submit_btn = await tab.find(
            tag_name='button',
            class_name='btn-primary',
            text='Submit'
        )
        # Find by ID
        username_field = await tab.find(id='username')
        # Find multiple elements
        all_links = await tab.find(tag_name='a', find_all=True)
        # CSS selectors and XPath
        nav_menu = await tab.query('nav.main-menu')
        specific_item = await tab.query('//div[@data-testid="item-123"]')
        # With timeout and error handling
        delayed_element = await tab.find(
            class_name='dynamic-content',
            timeout=10,
            raise_exc=False  # Returns None if not found
        )
        # Advanced: Custom attributes
        custom_element = await tab.find(
            data_testid='submit-button',
            aria_label='Submit form'
        )

asyncio.run(element_finding_examples())
```

### Browser-context HTTP requests - game changer for hybrid automation!
Ever wished you could make HTTP requests that automatically inherit all your browser's session state? **Now you can!**<br>
The `tab.request` property gives you a beautiful `requests`-like interface that executes HTTP calls directly in the browser's JavaScript context. This means every request automatically gets cookies, authentication headers, CORS policies, and session state, just as if the browser made the request itself.

**Perfect for Hybrid Automation:**
```python
# Navigate to a site and login normally with PyDoll
await tab.go_to('https://example.com/login')
await (await tab.find(id='username')).type_text('user@example.com')
await (await tab.find(id='password')).type_text('password')
await (await tab.find(id='login-btn')).click()

# Now make API calls that inherit the logged-in session!
response = await tab.request.get('https://example.com/api/user/profile')
user_data = response.json()

# POST data while staying authenticated
response = await tab.request.post(
    'https://example.com/api/settings', 
    json={'theme': 'dark', 'notifications': True}
)

# Access response content in different formats
raw_data = response.content
text_data = response.text
json_data = response.json()

# Check cookies that were set
for cookie in response.cookies:
    print(f"Cookie: {cookie['name']} = {cookie['value']}")

# Add custom headers to your requests
headers = [
    {'name': 'X-Custom-Header', 'value': 'my-value'},
    {'name': 'X-API-Version', 'value': '2.0'}
]

await tab.request.get('https://api.example.com/data', headers=headers)

```

### New expect_download() context manager — robust file downloads made easy!
Tired of fighting with flaky download flows, missing files, or racy event listeners? Meet `tab.expect_download()`, a delightful, reliable way to handle file downloads.

- Automatically sets the browser’s download behavior
- Works with your own directory or a temporary folder (auto-cleaned!)
- Waits for completion with a timeout (so your tests don’t hang)
- Gives you a handy handle to read bytes/base64 or check `file_path`

Tiny example that just works:

```python
import asyncio
from pathlib import Path
from pydoll.browser import Chrome

async def download_report():
    async with Chrome() as browser:
        tab = await browser.start()
        await tab.go_to('https://example.com/reports')

        target_dir = Path('/tmp/my-downloads')
        async with tab.expect_download(keep_file_at=target_dir, timeout=10) as download:
            # Trigger the download in the page (button/link/etc.)
            await (await tab.find(text='Download latest report')).click()
            # Wait until finished and read the content
            data = await download.read_bytes()
            print(f"Downloaded {len(data)} bytes to: {download.file_path}")

asyncio.run(download_report())
```

### Total browser control with custom preferences! (thanks to [@LucasAlvws](https://github.com/LucasAlvws))
Want to completely customize how Chrome behaves? **Now you can control EVERYTHING!**<br>
The new `browser_preferences` system gives you access to hundreds of internal Chrome settings that were previously impossible to change programmatically. We're talking about deep browser customization that goes way beyond command-line flags!

**The possibilities are endless:**
```python
options = ChromiumOptions()

# Create the perfect automation environment
options.browser_preferences = {
    'download': {
        'default_directory': '/tmp/downloads',
        'prompt_for_download': False,
        'directory_upgrade': True,
        'extensions_to_open': ''  # Don't auto-open any downloads
    },
    'profile': {
        'default_content_setting_values': {
            'notifications': 2,        # Block all notifications
            'geolocation': 2,         # Block location requests
            'media_stream_camera': 2, # Block camera access
            'media_stream_mic': 2,    # Block microphone access
            'popups': 1               # Allow popups (useful for automation)
        },
        'password_manager_enabled': False,  # Disable password prompts
        'exit_type': 'Normal'              # Always exit cleanly
    },
    'intl': {
        'accept_languages': 'en-US,en',
        'charset_default': 'UTF-8'
    },
    'browser': {
        'check_default_browser': False,    # Don't ask about default browser
        'show_update_promotion_infobar': False
    }
}

# Or use the convenient helper methods
options.set_default_download_directory('/tmp/downloads')
options.set_accept_languages('en-US,en,pt-BR')  
options.prompt_for_download = False
```

**Real-world power examples:**
- **Silent downloads** - No prompts, no dialogs, just automated downloads
- **Block ALL distractions** - Notifications, popups, camera requests, you name it
- **Perfect for CI/CD** - Disable update checks, default browser prompts, crash reporting
- **Multi-region testing** - Change languages, timezones, and locale settings instantly
- **Security hardening** - Lock down permissions and disable unnecessary features
- **Advanced fingerprinting control** - Modify browser install dates, engagement history, and behavioral patterns

**Fingerprint customization for stealth automation:**
```python
import time

# Simulate a browser that's been around for months
fake_engagement_time = int(time.time()) - (7 * 24 * 60 * 60)  # 7 days ago

options.browser_preferences = {
    'settings': {
        'touchpad': {
            'natural_scroll': True,
        }
    },
    'profile': {
        'last_engagement_time': fake_engagement_time,
        'exit_type': 'Normal',
        'exited_cleanly': True
    },
    'newtab_page_location_override': 'https://www.google.com',
    'session': {
        'restore_on_startup': 1,  # Restore last session
        'startup_urls': ['https://www.google.com']
    }
}
```

### Concurrent Automation

```python
import asyncio
from pydoll.browser import Chrome

async def scrape_page(url, tab):
    await tab.go_to(url)
    title = await tab.execute_script('return document.title')
    links = await tab.find(tag_name='a', find_all=True)
    return {
        'url': url,
        'title': title,
        'link_count': len(links)
    }

async def concurrent_scraping():
    browser = Chrome()
    tab_google = await browser.start()
    tab_duckduckgo = await browser.new_tab()
    tasks = [
        scrape_page('https://google.com/', tab_google),
        scrape_page('https://duckduckgo.com/', tab_duckduckgo)
    ]
    results = await asyncio.gather(*tasks)
    print(results)
    await browser.stop()

asyncio.run(concurrent_scraping())
```

[Back to Top](#pydoll-automate-the-web-naturally)

## 🔧 Quick Troubleshooting

*   **Browser Not Found:**  Specify the browser location:

    ```python
    from pydoll.browser import Chrome
    from pydoll.browser.options import ChromiumOptions

    options = ChromiumOptions()
    options.binary_location = '/path/to/your/chrome'
    browser = Chrome(options=options)
    ```

*   **Browser Starts After `FailedToStartBrowser` Error:** Increase the start timeout:

    ```python
    from pydoll.browser import Chrome
    from pydoll.browser.options import ChromiumOptions

    options = ChromiumOptions()
    options.start_timeout = 20  # default is 10 seconds

    browser = Chrome(options=options)
    ```

*   **Need a Proxy:** Configure proxy settings:

    ```python
    options.add_argument('--proxy-server=your-proxy:port')
    ```

*   **Running in Docker:** Add these arguments:

    ```python
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    ```

[Back to Top](#pydoll-automate-the-web-naturally)

## 📚 Documentation

For comprehensive information, visit the [official Pydoll documentation](https://pydoll.tech/) for:

*   Step-by-step tutorials
*   Complete API reference
*   Advanced techniques

[Back to Top](#pydoll-automate-the-web-naturally)

## 🤝 Contributing

Help improve Pydoll! Review the [contribution guidelines](CONTRIBUTING.md) and submit your contributions. All contributions are welcome!

[Back to Top](#pydoll-automate-the-web-naturally)

## 💖 Support

Show your support! Consider [sponsoring the project on GitHub](https://github.com/sponsors/thalissonvs). Other ways to support:

*   Star the repository
*   Share on social media
*   Write tutorials
*   Report issues

[Back to Top](#pydoll-automate-the-web-naturally)

## 💬 Spread the Word

If Pydoll helped you, share it with others!

[Back to Top](#pydoll-automate-the-web-naturally)

## 📄 License

Pydoll is licensed under the [MIT License](LICENSE).

[Back to Top](#pydoll-automate-the-web-naturally)