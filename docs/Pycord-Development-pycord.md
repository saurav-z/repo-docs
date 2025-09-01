[![Pycord](https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png)](https://github.com/Pycord-Development/pycord)

# Pycord: The Modern Python Library for Discord Bots

**Pycord is a powerful and easy-to-use Python library, enabling you to build feature-rich and asynchronous Discord bots with ease.**

[![PyPI Version](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest Release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord Server](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Pycord-Development?style=for-the-badge)](https://github.com/sponsors/Pycord-Development)
[![Crowdin Translations](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

## Key Features

*   **Modern and Pythonic:** Built with `async` and `await` for efficient asynchronous programming.
*   **Robust Rate Limit Handling:** Automatically manages Discord's rate limits to ensure your bot runs smoothly.
*   **Optimized Performance:** Designed for both speed and efficient memory usage.
*   **Full Application API Support:** Access to the complete Discord API.

## Supported Python Versions

Pycord supports Python versions 3.9 through 3.13.

## Installation

**Requires Python 3.9 or higher.**

To install Pycord:

```bash
# Without voice support (Linux/macOS)
python3 -m pip install -U py-cord

# Without voice support (Windows)
py -3 -m pip install -U py-cord

# With full voice support (Linux/macOS)
python3 -m pip install -U "py-cord[voice]"

# With full voice support (Windows)
py -3 -m pip install -U py-cord[voice]

# To install additional packages for speedup (Linux/macOS)
python3 -m pip install -U "py-cord[speed]"

# To install additional packages for speedup (Windows)
py -3 -m pip install -U py-cord[speed]
```

### Installing the Development Version

```bash
# Clone the repository
git clone https://github.com/Pycord-Development/pycord
cd pycord

# Install with voice support
python3 -m pip install -U .[voice]
```

Or, without cloning:

```bash
# (Linux/macOS)
python3 -m pip install git+https://github.com/Pycord-Development/pycord

# (Windows)
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

## Optional Packages

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`__, `brotlipy <https://pypi.org/project/brotlipy/>`__, `cchardet <https://pypi.org/project/cchardet/>`__ (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>`__ (for json speedup)

**Note for Linux voice support:** You must install the following packages via your system's package manager *before* installing Pycord with voice support:

*   libffi-dev (or libffi-devel on some systems)
*   python-dev (e.g., python3.10-dev for Python 3.10)

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

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Learn how to create Discord bots with Pycord](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)
*   [Pycord GitHub Repository](https://github.com/Pycord-Development/pycord)

## Translations

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)