<!-- Improved README for Pycord -->

<div align="center">
  <a href="https://github.com/Pycord-Development/pycord">
    <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord Logo" width="200">
  </a>
  <br>
  <a href="https://github.com/Pycord-Development/pycord">Pycord</a> - The modern, feature-rich, and asynchronous Python library for building powerful Discord bots.
  <br>
  <!-- Badges - Consider moving these lower down for better readability -->
  <p>
    <a href="https://pypi.org/project/py-cord/">
      <img src="https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white" alt="PyPI Version">
    </a>
    <a href="https://pypi.python.org/pypi/py-cord">
      <img src="https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Versions">
    </a>
    <a href="https://pypi.python.org/pypi/py-cord">
      <img src="https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge" alt="PyPI Downloads">
    </a>
    <a href="https://github.com/Pycord-Development/pycord/releases">
      <img src="https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white" alt="Latest Release">
    </a>
    <a href="https://pycord.dev/discord">
      <img src="https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white" alt="Discord Server">
    </a>
    <a href="https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on">
      <img src="https://badges.crowdin.net/badge/dark/crowdin-on-light.png" alt="Crowdin">
    </a>
  </p>
</div>

## About Pycord

Pycord is a robust and user-friendly Python library, designed to make creating Discord bots easier and more enjoyable. It offers a modern and efficient way to interact with the Discord API, allowing developers to build bots with powerful features and a great user experience.

## Key Features

*   **Asynchronous and Modern:** Built with `async` and `await` for efficient, non-blocking operations.
*   **Rate Limit Handling:** Automatically manages Discord's rate limits, preventing your bot from getting blocked.
*   **Optimized Performance:** Designed for speed and efficient memory usage to handle large bot deployments.
*   **Full API Coverage:** Supports the complete Discord application API, providing access to all features.

## Installation

**Prerequisites:** Python 3.9 or higher is required.

To install Pycord, choose one of the options below.

**Install without Voice Support:**

```bash
# Linux/macOS
python3 -m pip install -U py-cord

# Windows
py -3 -m pip install -U py-cord
```

**Install with Voice Support:**

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[voice]"

# Windows
py -3 -m pip install -U py-cord[voice]
```

**Install for Speed Optimization:**

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"

# Windows
py -3 -m pip install -U py-cord[speed]
```

**Install the Development Version:**

```bash
# Clone the repository
git clone https://github.com/Pycord-Development/pycord
cd pycord
python3 -m pip install -U .[voice]

# Alternatively, install directly from GitHub:
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord

# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

**Important: Linux Voice Support Dependencies**

When installing voice support on Linux, ensure you have the necessary packages installed *before* running the install commands:

*   `libffi-dev` (or `libffi-devel` on some systems)
*   `python-dev` (e.g., `python3.10-dev` for Python 3.10)

## Optional Packages

For enhanced functionality and performance, consider installing these optional packages:

*   `PyNaCl` (for voice support)
*   `aiodns`, `brotlipy`, `cchardet` (for aiohttp speedup)
*   `msgspec` (for JSON speedup)

## Quick Start Example

Get up and running with a simple slash command bot:

```python
import discord

bot = discord.Bot()

@bot.slash_command()
async def hello(ctx, name: str = None):
    name = name or ctx.author.name
    await ctx.respond(f"Hello {name}!")

@bot.user_command(name="Say Hello")
async def hi(ctx, user):
    await ctx.respond(f"{ctx.author.mention} says hello to {user.name}!")

bot.run("YOUR_BOT_TOKEN") # Replace with your bot token
```

## Traditional Commands Example

If you prefer traditional commands, here's an example:

```python
import discord
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=">", intents=intents)

@bot.command()
async def ping(ctx):
    await ctx.send("pong")

bot.run("YOUR_BOT_TOKEN") # Replace with your bot token
```

**Important Note:**  Never share your bot token.  It is used to authenticate your bot and grant access to your Discord account.

## Useful Links

*   **Documentation:** [https://docs.pycord.dev/en/master/index.html](https://docs.pycord.dev/en/master/index.html)
*   **Getting Started Guide:** [https://guide.pycord.dev](https://guide.pycord.dev)
*   **Discord Server:** [https://pycord.dev/discord](https://pycord.dev/discord)
*   **Official Discord Developers Server:** [https://discord.gg/discord-developers](https://discord.gg/discord-developers)
*   **Source Code:** [GitHub Repository](https://github.com/Pycord-Development/pycord)