<div align="center">
  <a href="https://github.com/Pycord-Development/pycord">
    <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord Logo" width="200">
  </a>
  <h1>Pycord: The Modern Python Library for Building Amazing Discord Bots</h1>
</div>

Pycord is a powerful, easy-to-use, and feature-rich Python library, providing an asynchronous API wrapper for the Discord platform.

[![PyPI version info](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI supported Python versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord server invite](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Pycord-Development?style=for-the-badge)](https://github.com/sponsors/Pycord-Development)
[![Crowdin | Agile localization for tech companies](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

## Key Features

*   **Modern Pythonic API:** Built with `async` and `await` for efficient, non-blocking operations.
*   **Robust Rate Limit Handling:** Automatically manages Discord's rate limits, ensuring your bot runs smoothly.
*   **Optimized Performance:** Designed for both speed and efficient memory usage.
*   **Full Application API Support:** Provides access to the complete Discord API for rich bot functionality.
*   **Easy to Use:** Provides both slash command and traditional command support.

## Getting Started

### Requirements

*   Python 3.9 - 3.13

### Installation

Install the core library:

```bash
# Linux/macOS
python3 -m pip install -U py-cord
# Windows
py -3 -m pip install -U py-cord
```

Install with Voice Support:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[voice]"
# Windows
py -3 -m pip install -U py-cord[voice]
```

Install with Speed Optimizations:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
py -3 -m pip install -U py-cord[speed]
```

Install from Source:

```bash
$ git clone https://github.com/Pycord-Development/pycord
$ cd pycord
$ python3 -m pip install -U .[voice]
```

Or:

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord
# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

**Note for Linux Voice Support:** Before installing with voice support on Linux, ensure you have the necessary development packages installed.  For example, use your system's package manager to install `libffi-dev` (or `libffi-devel`) and `python3.10-dev` (or the appropriate Python version).

### Optional Packages

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`__, `brotlipy <https://pypi.org/project/brotlipy/>`__, `cchardet <https://pypi.org/project/cchardet/>`__ (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>`__ (for json speedup)

## Quick Example

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

bot.run("token")
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

bot.run("token")
```

**Important:** Always keep your bot token secure and never share it publicly.

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Guide: Learn how to create Discord bots with Pycord](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)
*   [Pycord GitHub Repository](https://github.com/Pycord-Development/pycord)

## Translations

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)