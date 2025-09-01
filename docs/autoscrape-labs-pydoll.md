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

## Automate Web Interactions Effortlessly with Pydoll

**Pydoll is a Python library that simplifies web automation by directly connecting to the Chrome DevTools Protocol, eliminating the need for external drivers and making automation more natural and user-friendly.**  Visit the [Pydoll GitHub Repository](https://github.com/autoscrape-labs/pydoll) for the latest updates.

**Key Features:**

*   ✅ **Zero Webdriver Dependencies:**  Avoid compatibility headaches.
*   🤖 **Human-Like Interaction Engine:**  Bypass behavioral CAPTCHAs.
*   🚀 **Asynchronous Performance:**  Run automations at high speed.
*   🖱️ **Humanized Interactions:** Mimic real user behavior for more reliable automation.
*   ✨ **Simplicity:** Easy to install and start automating.
*   🌐 **Remote Browser Control:** Connect and control Chrome instances via WebSocket.
*   🧭 **Advanced DOM Traversal:**  `get_children_elements()` and `get_siblings_elements()` for efficient DOM navigation.
*   🚦 **WebElement State Waiting:** `wait_until(...)` methods for reliable element state validation.
*   🔗 **Browser-Context HTTP Requests:**  Seamlessly integrate with API calls within your automation.
*   ⬇️ **Robust File Downloads:** Simplify file download handling with `expect_download()`.
*   ⚙️ **Custom Browser Preferences:**  Fine-tune Chrome behavior with detailed preferences.
*   💻 **Concurrent Automation:**  Execute multiple tasks simultaneously for faster results.

## What's New

### Remote Connections via WebSocket

Control any Chrome instance remotely using its WebSocket address. This feature simplifies automation against remote browsers, CI environments, containers, and shared debugging targets.

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

Simplify the traversal of complex layouts.

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

*   New `wait_until(...)` on `WebElement` to await element states with minimal code:

```python
# Wait until it becomes visible OR the timeout expires
await element.wait_until(is_visible=True, timeout=5)

# Wait until it becomes interactable (visible, on top, receiving pointer events)
await element.wait_until(is_interactable=True, timeout=10)
```

*   Methods now public on `WebElement`:
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

## Installation

Install Pydoll using pip:

```bash
pip install pydoll-python
```

## Getting Started

### Simple Google Search Example

Here's how to perform a Google search and click on the first result:

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

### Data Extraction Example

Extract data from a webpage.

```python
import asyncio

from pydoll.browser import Chrome
from pydoll.constants import Key

async def extract_data():
    async with Chrome() as browser:
        tab = await browser.start()
        await tab.go_to('https://github.com/autoscrape-labs/pydoll')
         description = await (await tab.query(
            '//h2[contains(text(), "About")]/following-sibling::p',
            timeout=10,
        )).text
        number_of_stars = await (await tab.find(
            id='repo-stars-counter-star'
        )).text

        number_of_forks = await (await tab.find(
            id='repo-network-counter'
        )).text
        number_of_issues = await (await tab.find(
            id='issues-repo-tab-count',
        )).text
        number_of_pull_requests = await (await tab.find(
            id='pull-requests-repo-tab-count',
        )).text

        data = {
            'description': description,
            'number_of_stars': number_of_stars,
            'number_of_forks': number_of_forks,
            'number_of_issues': number_of_issues,
            'number_of_pull_requests': number_of_pull_requests,
        }
        print(data)
asyncio.run(extract_data())
```

### Custom Configurations

Configure your browser with custom settings.

```python
from pydoll.browser import Chrome
from pydoll.browser.options import ChromiumOptions as Options

async def custom_automation():
    # Configure browser options
    options = Options()
    options.add_argument('--proxy-server=username:password@ip:port')
    options.add_argument('--window-size=1920,1080')
    options.binary_location = '/path/to/your/browser'
    options.start_timeout = 20

    async with Chrome(options=options) as browser:
        tab = await browser.start()
        # Your automation code here
        await tab.go_to('https://example.com')
        # The browser is now using your custom settings

asyncio.run(custom_automation())
```

## Advanced Features

### Advanced Element Search

Find elements using various methods:

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

### Browser-context HTTP requests

Make HTTP requests that inherit your browser's session state.

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

### New expect_download() context manager

Easily handle file downloads:

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

### Total browser control with custom preferences

Customize Chrome's behavior:

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

### Concurrent Automation

Run multiple tasks simultaneously:

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

## Troubleshooting

### Common Issues

*   **Browser Not Found:**

    ```python
    from pydoll.browser import Chrome
    from pydoll.browser.options import ChromiumOptions

    options = ChromiumOptions()
    options.binary_location = '/path/to/your/chrome'
    browser = Chrome(options=options)
    ```

*   **Browser Fails to Start:**

    ```python
    from pydoll.browser import Chrome
    from pydoll.browser.options import ChromiumOptions

    options = ChromiumOptions()
    options.start_timeout = 20  # default is 10 seconds

    browser = Chrome(options=options)
    ```

*   **Using a Proxy:**

    ```python
    options.add_argument('--proxy-server=your-proxy:port')
    ```

*   **Running in Docker:**

    ```python
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    ```

## Documentation

For comprehensive documentation, examples, and API details, please visit the [official documentation](https://pydoll.tech/).

## Contributing

Contributions are welcome! Refer to our [contribution guidelines](CONTRIBUTING.md).

## Support My Work

If you find Pydoll useful, consider supporting me on [GitHub Sponsors](https://github.com/sponsors/thalissonvs).

## Spread the Word

If Pydoll saved you time, mental health, or a keyboard, give it a ⭐, share it, or tell your dev friends!

## License

Pydoll is licensed under the [MIT License](LICENSE).

<p align="center">
  <b>Pydoll</b> — Making browser automation magical!
</p>