# Pycord: The Modern Python Library for Discord Bots

**Pycord** is a powerful and user-friendly Python library designed to help you build feature-rich and efficient Discord bots.  Get started today with this easy-to-use wrapper!  [Visit the original repository on GitHub](https://github.com/Pycord-Development/pycord).

[![PyPI Version](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Supported Python Versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest Release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord Server Invite](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Pycord-Development?style=for-the-badge)](https://github.com/sponsors/Pycord-Development)
[![Crowdin](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

## Key Features

*   **Modern Pythonic API:** Leveraging `async` and `await` for efficient asynchronous operations.
*   **Robust Rate Limit Handling:** Automatically manages and handles Discord's rate limits.
*   **Optimized Performance:** Designed for speed and minimal memory usage.
*   **Full Application API Support:** Access to all features of the Discord API.

## Installation

**Prerequisites:** Python 3.9 or higher is required.

Install the basic package:

```bash
# Linux/macOS
python3 -m pip install -U py-cord

# Windows
py -3 -m pip install -U py-cord
```

Install with voice support:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[voice]"

# Windows
py -3 -m pip install -U py-cord[voice]
```

Install for additional speedup:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
py -3 -m pip install -U py-cord[speed]
```

Install the development version:

```bash
$ git clone https://github.com/Pycord-Development/pycord
$ cd pycord
$ python3 -m pip install -U .[voice]
```

or without cloning the repository:

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord
# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

### Optional Packages

*   `PyNaCl <https://pypi.org/project/PyNaCl/>` (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`, `brotlipy <https://pypi.org/project/brotlipy/>`, `cchardet <https://pypi.org/project/cchardet/>` (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>` (for json speedup)

**Note for Linux Voice Support:** Ensure you have these packages installed via your system's package manager *before* installing Pycord with voice:

*   `libffi-dev` (or `libffi-devel`)
*   `python-dev` (e.g., `python3.10-dev`)

## Quickstart Example

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

bot.run("YOUR_BOT_TOKEN")  # Replace with your bot token
```

## Traditional Commands Example

```python
import discord
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=">", intents=intents)

@bot.command()
async def ping(ctx):
    await ctx.send("pong")

bot.run("YOUR_BOT_TOKEN")  # Replace with your bot token
```

**Important:**  Always keep your bot token secure!  Never share it publicly.

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Learn how to create Discord bots with Pycord](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)

## Translations

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)