<!-- Add a title tag to improve SEO -->
<title>Browser Use: AI-Powered Browser Automation | Automate Web Tasks</title>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Browser Use Logo - AI controlling a browser" src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Browser Use: Automate Your Browser with AI</h1>

[![Docs](https://img.shields.io/badge/Docs-📕-blue?style=for-the-badge)](https://docs.browser-use.com)
[![Browser-use cloud](https://img.shields.io/badge/Browser_Use_Cloud-☁️-blue?style=for-the-badge&logo=rocket&logoColor=white)](https://cloud.browser-use.com)

[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)
[![Twitter Follow](https://img.shields/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
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

## 🌟 **Introducing Browser Use: The AI-Powered Browser Automation Tool**

Browser Use empowers you to control your browser with natural language, automating complex web tasks with ease. [Check out the original repo](https://github.com/browser-use/browser-use).

**Key Features:**

*   **Natural Language Control:** Interact with your browser using simple, human-readable commands.
*   **AI-Driven Automation:** Leverage the power of AI to automate web tasks such as data extraction, form filling, and web application interaction.
*   **Cloud Integration:** Access a cloud-based browser for increased reliability and scalability.
*   **MCP Integration:** Seamlessly integrate with Claude Desktop for enhanced browser automation capabilities.
*   **Easy Setup:** Quick and straightforward installation using `uv pip install browser-use`.

## 🚀 Quickstart

```bash
# Install the package
uv pip install browser-use

# Download chromium using playwright's shortcut
uvx playwright install chromium --with-deps --no-shell

# Set your API Key
# Create a .env file and add your API key. Don't have one? Start with a [free Gemini key](https://aistudio.google.com/app/u/1/apikey?pli=1).
#  GEMINI_API_KEY=
# Run your first agent:
```

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

Explore the [library docs](https://docs.browser-use.com) and [cloud docs](https://docs.cloud.browser-use.com) for further customization.

## 🎬 Demos

### Grocery Shopping Demo
[Task](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py): Add grocery items to cart, and checkout.

[![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

<br/><br/>

### Job Application Demo
[Task](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py): Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs, if you need help, ask me.

https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

<br/><br/>

See [more examples](https://docs.browser-use.com/examples) and give us a star!

## 💻 MCP Integration

Integrate with Claude Desktop for advanced browser automation. See the [MCP docs](https://docs.browser-use.com/customize/mcp-server).

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

**Effortlessly automate your web interactions with the power of AI.**

<img src="https://github.com/user-attachments/assets/06fa3078-8461-4560-b434-445510c1766f" width="400"/>

[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/intent/user?screen_name=mamagnus00)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/intent/user?screen_name=gregpr07)

</div>

<div align="center">
Made with ❤️ in Zurich and San Francisco
 </div>