<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser-Use: Automate Your Browser with AI</h1>

**Browser-Use empowers you to control your browser with natural language, unlocking a new level of automation.** Learn more at the [original repository](https://github.com/browser-use/browser-use).

[![Docs](https://img.shields.io/badge/Docs-📕-blue?style=for-the-badge)](https://docs.browser-use.com)
[![Browser-use cloud](https://img.shields.io/badge/Browser_Use_Cloud-☁️-blue?style=for-the-badge&logo=rocket&logoColor=white)](https://cloud.browser-use.com)

[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Merch store](https://img.shields.io/badge/Merch_store-👕-blue)](https://browsermerch.com)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

<!-- Keep these links. Translations will automatically update with the README. -->
[Deutsch](https://www.readme-i18n.com/browser-use/browser-use?lang=de) |
[Español](https://www.readme-i18n.com/browser-use/browser-use?lang=es) |
[français](https://www.readme-i18n.com/browser-use/browser-use?lang=fr) |
[日本語](https://www.readme-i18n.com/browser-use/browser-use?lang=ja) |
[한국어](https://www.readme-i18n.com/browser-use/browser-use?lang=ko) |
[Português](https://www.readme-i18n.com/browser-use/browser-use?lang=pt) |
[Русский](https://www.readme-i18n.com/browser-use/browser-use?lang=ru) |
[中文](https://www.readme-i18n.com/browser-use/browser-use?lang=zh)

## Key Features

*   **AI-Powered Browser Automation:** Control your browser with natural language prompts.
*   **Easy Setup:** Get started quickly with simple installation steps.
*   **Cloud Integration:** Leverage the Browser-Use Cloud for enhanced performance.
*   **Example Use Cases:** Explore practical demos such as grocery shopping and job applications.
*   **MCP Integration:** Seamlessly integrate with MCP for extended capabilities.

## Quickstart

Follow these steps to get up and running:

**1. Install Dependencies:**

```bash
# Use uv (Python>=3.11)
uv pip install browser-use
```

**2. Download Chromium:**

```bash
uvx playwright install chromium --with-deps --no-shell
```

**3. Configure API Key:**

Create a `.env` file and add your API key (e.g., a free [Gemini key](https://aistudio.google.com/app/u/1/apikey?pli=1)):

```
GEMINI_API_KEY=YOUR_API_KEY
```

**4. Run Your First Agent:**

```python
from browser_use import Agent, ChatGoogle
from dotenv import load_dotenv
load_dotenv()

agent = Agent(
    task="Find the number of stars of the browser-use repo",
    llm=ChatGoogle(model="gemini-2.5-flash"),
    # browser=Browser(use_cloud=True),  # Uses Browser-Use cloud for the browser
)
agent.run_sync()
```

**5. Explore Further:**

Check out the [library docs](https://docs.browser-use.com) and [cloud docs](https://docs.cloud.browser-use.com) for more detailed information.

## Demos & Use Cases

Explore the possibilities with these examples:

**1. Automated Grocery Shopping**

*   [Task](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py): Add grocery items to cart and checkout.
    [![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

**2. AI-Powered Job Application**

*   [Task](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py): Read your CV, find relevant ML jobs, and apply for them.
    <br/>
    https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04
    <br/><br/>
    For more examples, visit the [examples](https://docs.browser-use.com/examples) page and give us a star!

## MCP Integration

Integrate Browser-Use with Claude Desktop for extended browser automation functionalities, including web scraping and form filling.  See the [MCP docs](https://docs.browser-use.com/customize/mcp-server).

```json
{
  "mcpServers": {
    "browser-use": {
      "command": "uvx",
      "args": ["browser-use[cli]", "--mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

<div align="center">
  
**Unleash the power of AI to control your digital world, making your browser your personal assistant.**

<img src="https://github.com/user-attachments/assets/06fa3078-8461-4560-b434-445510c1766f" width="400"/>

[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)

</div>

<div align="center">
Made with ❤️ in Zurich and San Francisco
 </div>