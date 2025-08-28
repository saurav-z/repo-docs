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


## 🚀 Automate Web Interactions with Ease Using Pydoll

Tired of struggling with web automation?  **Pydoll simplifies web automation by connecting directly to the Chrome DevTools Protocol, offering a user-friendly and powerful alternative to traditional WebDriver-based approaches.**  Focus on your automation logic, not complex configurations.

### Key Features:

*   ✅ **Zero Webdriver Dependencies:**  Say goodbye to driver compatibility headaches.
*   🤖 **Human-like Interaction:** Mimics real user behavior, including advanced CAPTCHA bypass.
*   ⚡ **Asynchronous Performance:** Enables high-speed automation and concurrent tasks.
*   🖱️ **Intuitive Interaction Engine:** Provides realistic clicking, navigation, and element interaction.
*   ✨ **Simplicity:** Get started automating quickly with a straightforward installation process.
*   🌐 **Remote Browser Control:** Connect to and control any Chrome instance via WebSocket.

[Explore the Pydoll repository for more details.](https://github.com/autoscrape-labs/pydoll)

## ✨ What's New

### Remote Connections via WebSocket: Control any Chrome from anywhere!

You can now connect to an already running browser remotely via its WebSocket address and use the full Pydoll API immediately.

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
  - `is_interactable()`
  - `is_on_top()`
  - `execute_script(script: str, return_by_value: bool = False)`

```python
# Visually outline the element via JS
await element.execute_script("this.style.outline='2px solid #22d3ee'")

# Confirm states
visible = await element.is_visible()
interactable = await element.is_interactable()
on_top = await element.is_on_top()
```

These additions simplify waiting and state validation before clicking/typing, reducing flakiness and making automations more predictable.

## 📦 Installation

```bash
pip install pydoll-python
```

## 🚀 Getting Started

### Your first automation

Let's start with a real example: an automation that performs a Google search and clicks on the first result. With this example, we can see how the library works and how you can start automating your tasks.

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

Okay, now let's see how we can extract data from a page, using the same previous example.
Let's consider in the code below that we're already on the Pydoll page. We want to extract the following information:

- Project description
- Number of stars
- Number of forks
- Number of issues
- Number of pull requests

Let's get started! To get the project description, we'll use xpath queries. You can check the documentation on how to build your own queries.

```python
description = await (await tab.query(
    '//h2[contains(text(), "About")]/following-sibling::p',
    timeout=10,
)).text
```

And that's it! Let's understand what this query does:

1. `//h2[contains(text(), "About")]` - Selects the first `<h2>` that contains "About"
2. `/following-sibling::p` - Selects the first `<p>` that comes after the `<h2>`

Now let's get the rest of the data:

```python
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

```

We managed to extract all the necessary data!

### Custom Configurations

Sometimes we need more control over the browser. Pydoll offers a flexible way to do this. Let's see the example below:


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

In this example, we're configuring the browser to use a proxy and a 1920x1080 window, in addition to a custom path for the Chrome binary, in case your installation location is different from the common defaults.


## ⚡ Advanced Features

Pydoll provides powerful features for advanced users.

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

## 🔧 Quick Troubleshooting

**Browser not found?**
```python
from pydoll.browser import Chrome
from pydoll.browser.options import ChromiumOptions

options = ChromiumOptions()
options.binary_location = '/path/to/your/chrome'
browser = Chrome(options=options)
```

**Browser starts after a FailedToStartBrowser error?**
```python
from pydoll.browser import Chrome
from pydoll.browser.options import ChromiumOptions

options = ChromiumOptions()
options.start_timeout = 20  # default is 10 seconds

browser = Chrome(options=options)
```

**Need a proxy?**
```python
options.add_argument('--proxy-server=your-proxy:port')
```

**Running in Docker?**
```python
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
```

## 📚 Documentation

For complete documentation, detailed examples and deep dives into all Pydoll functionalities, visit our [official documentation](https://pydoll.tech/).

The documentation includes:
- **Getting Started Guide** - Step-by-step tutorials
- **API Reference** - Complete method documentation
- **Advanced Techniques** - Network interception, event handling, performance optimization

>The chinese version of this README is [here](README_zh.md).

## 🤝 Contributing

We welcome contributions to make Pydoll even better!  Refer to our [contribution guidelines](CONTRIBUTING.md) to get started.

## 💖 Support My Work

Show your support by sponsoring on [GitHub](https://github.com/sponsors/thalissonvs).

## 💬 Spread the word

If Pydoll saved you time or mental health, give it a ⭐, share it, or tell your friends.

## 📄 License

Pydoll is licensed under the [MIT License](LICENSE).

<p align="center">
  <b>Pydoll</b> — Making browser automation magical!
</p>