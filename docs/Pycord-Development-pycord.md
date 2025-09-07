<!-- Pycord Header -->
<div align="center">
  <a href="https://github.com/Pycord-Development/pycord">
    <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord v3" width="200"/>
  </a>
  <br/>
  <a href="https://github.com/Pycord-Development/pycord">
    <img src="https://img.shields.io/github/stars/Pycord-Development/pycord?style=social" alt="Github Stars"/>
  </a>
</div>

# Pycord: The Modern Python Library for Discord Bots

**Pycord is your go-to library for building powerful and feature-rich Discord bots in Python, offering an easy-to-use, asynchronous API.**

[![PyPI version info](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI supported Python versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest Release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord server invite](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Pycord-Development?style=for-the-badge)](https://github.com/sponsors/Pycord-Development)
[![Crowdin | Agile localization for tech companies](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

## Key Features

*   **Modern Pythonic API:** Built with `async` and `await` for efficient asynchronous programming.
*   **Robust Rate Limit Handling:** Automatically handles Discord's rate limits, ensuring your bot runs smoothly.
*   **Optimized Performance:** Designed for both speed and minimal memory usage.
*   **Full Application API Support:** Access all of Discord's features through a comprehensive API.
*   **Easy to Learn:** Designed with a clean API to create Discord Bots

## Installation

**Requires Python 3.9 or higher.**

To install Pycord without voice support:

```bash
# Linux/macOS
python3 -m pip install -U py-cord

# Windows
py -3 -m pip install -U py-cord
```

To install Pycord with full voice support:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[voice]"

# Windows
py -3 -m pip install -U py-cord[voice]
```

To install optional speedup packages:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"

# Windows
py -3 -m pip install -U py-cord[speed]
```

To install the development version:

```bash
git clone https://github.com/Pycord-Development/pycord
cd pycord
python3 -m pip install -U .[voice]
```

or using pip:

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord

# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

### Optional Dependencies

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`_ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`, `brotlipy <https://pypi.org/project/brotlipy/>`, `cchardet <https://pypi.org/project/cchardet/>` (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>` (for json speedup)

**Linux Voice Support Note:** Before installing voice support, ensure you have installed the following packages using your system's package manager (e.g., `apt`, `dnf`):

*   `libffi-dev` (or `libffi-devel`)
*   `python-dev` (e.g., `python3.10-dev` for Python 3.10)

## Quick Start Example

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

bot.run("YOUR_BOT_TOKEN")
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

bot.run("YOUR_BOT_TOKEN")
```

**Important:**  Protect your bot token!  Never share it publicly, as it grants access to your bot.

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Pycord Guide - Learn how to create Discord bots with Pycord](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)
*   [Pycord GitHub Repository](https://github.com/Pycord-Development/pycord)

## Translations

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)