<!-- README.md -->

![Pycord v3](https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png)

# Pycord: The Modern Python Library for Discord Bots

**Pycord is a powerful, easy-to-use, and feature-rich asynchronous Python library, making it the go-to choice for building sophisticated Discord bots.**

[Visit the original repository on GitHub](https://github.com/Pycord-Development/pycord)

[![PyPI version info](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI supported Python versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord server invite](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![Crowdin | Agile localization for tech companies](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

## Key Features

*   **Asynchronous and Pythonic:** Leverages `async` and `await` for efficient and modern code.
*   **Rate Limit Handling:** Automatically handles Discord API rate limits, simplifying bot development.
*   **Optimized Performance:** Designed for both speed and efficient memory usage.
*   **Full Application API Support:** Provides comprehensive access to all Discord API features.
*   **Easy to use**: With a simple and intuitive API, creating a bot is easier than ever.

## Supported Python Versions

Pycord supports Python versions `3.9` through `3.13`.

## Installation

**Prerequisites:** Python 3.9 or higher is required.

**Install without voice support:**

```bash
# Linux/macOS
python3 -m pip install -U py-cord

# Windows
py -3 -m pip install -U py-cord
```

**Install with full voice support:**

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[voice]"

# Windows
py -3 -m pip install -U py-cord[voice]
```

**Install for speedup (optional):**

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
py -3 -m pip install -U py-cord[speed]
```

**Install the development version:**

```bash
$ git clone https://github.com/Pycord-Development/pycord
$ cd pycord
$ python3 -m pip install -U .[voice]
```

or:

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord
# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

### Optional Packages for Performance

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`, `brotlipy <https://pypi.org/project/brotlipy/>`, `cchardet <https://pypi.org/project/cchardet/>` (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>` (for json speedup)

**Important for Linux voice support:** Ensure you have installed the required system packages using your package manager before installing Pycord:

*   `libffi-dev` (or `libffi-devel` on some systems)
*   `python-dev` (e.g., `python3.10-dev` for Python 3.10)

## Quick Examples

**Slash Command Example:**

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

**Traditional Commands Example:**

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

Find more code examples in the `examples` directory.

**Security Note:**  Never share your bot token.  It grants full control over your bot.

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Guide: Learn how to create Discord bots with Pycord](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)

## Translations

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)