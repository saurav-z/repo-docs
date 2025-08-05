<!-- Improved README.md -->

# Pycord: The Modern Python Library for Discord Bots

**Pycord** is a powerful and easy-to-use Python library that simplifies building feature-rich and asynchronous Discord bots. ([View on GitHub](https://github.com/Pycord-Development/pycord))

[![PyPI version info](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI supported Python versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord server invite](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![Crowdin | Agile localization for tech companies](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

## Key Features

*   **Modern Pythonic API:** Leverage the power of `async` and `await` for efficient and responsive bot development.
*   **Robust Rate Limit Handling:** Pycord automatically handles Discord's rate limits, ensuring your bot runs smoothly.
*   **Optimized Performance:** Designed for both speed and efficient memory usage, maximizing your bot's performance.
*   **Full Application API Support:** Access the complete Discord API to build advanced and interactive bots.

## Installation

**Requires Python 3.9 - 3.13**

Install the library using `pip`:

```bash
# Without voice support (Linux/macOS)
python3 -m pip install -U py-cord
# Without voice support (Windows)
py -3 -m pip install -U py-cord

# With full voice support (Linux/macOS)
python3 -m pip install -U "py-cord[voice]"
# With full voice support (Windows)
py -3 -m pip install -U py-cord[voice]

# For aiohttp and json speedups (Linux/macOS)
python3 -m pip install -U "py-cord[speed]"
# For aiohttp and json speedups (Windows)
py -3 -m pip install -U py-cord[speed]
```

To install the development version:

```bash
$ git clone https://github.com/Pycord-Development/pycord
$ cd pycord
$ python3 -m pip install -U .[voice]
```

or

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord
# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

**Important Notes for Voice Support on Linux:** Ensure you have the following packages installed before installing Pycord with voice support:

*   `libffi-dev` (or `libffi-devel` on some systems)
*   `python-dev` (e.g., `python3.10-dev` for Python 3.10)

## Optional Packages

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`__, `brotlipy <https://pypi.org/project/brotlipy/>`__, `cchardet <https://pypi.org/project/cchardet/>`__ (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>`__ (for json speedup)

## Quickstart Examples

### Slash Commands

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

### Traditional Commands

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

**Important:** Replace `"YOUR_BOT_TOKEN"` with your actual bot token.  Keep your bot token secure!

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Guide](https://guide.pycord.dev) - Learn how to create Discord bots with Pycord
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)

## Translations

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/)