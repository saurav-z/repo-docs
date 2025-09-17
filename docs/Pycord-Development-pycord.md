<div align="center">
  <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord v3" width="200">
  <h1>Pycord: The Modern Discord API Wrapper for Python</h1>
</div>

**Pycord** is a powerful and user-friendly Python library designed to simplify the creation of Discord bots, offering a feature-rich and asynchronous-ready experience. ([View on GitHub](https://github.com/Pycord-Development/pycord))

[![PyPI Version](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Supported Python Versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest Release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord Server Invite](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Pycord-Development?style=for-the-badge)](https://github.com/sponsors/Pycord-Development)
[![Crowdin Translation Status](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

## Key Features

*   **Modern Pythonic API:** Leverages `async` and `await` for efficient and responsive bot development.
*   **Robust Rate Limit Handling:** Ensures your bot operates smoothly within Discord's API limits.
*   **Optimized Performance:** Designed for both speed and memory efficiency.
*   **Full Application API Support:** Access all the latest Discord features.

## Installation

**Requires Python 3.9 - 3.13**

Install the core library:

```bash
# Linux/macOS
python3 -m pip install -U py-cord

# Windows
py -3 -m pip install -U py-cord
```

For voice support, install with the `voice` extra:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[voice]"

# Windows
py -3 -m pip install -U py-cord[voice]
```

To install speedup packages:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
py -3 -m pip install -U py-cord[speed]
```

### Installing the Development Version

```bash
$ git clone https://github.com/Pycord-Development/pycord
$ cd pycord
$ python3 -m pip install -U .[voice]
```

or, without cloning:

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord
# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

### Optional Packages

*   `PyNaCl` (for voice support)
*   `aiodns`, `brotlipy`, `cchardet` (for aiohttp speedup)
*   `msgspec` (for json speedup)

**Important for Linux Voice Support:**  Before installing `py-cord[voice]`, ensure you have the following packages installed via your system's package manager:

*   `libffi-dev` (or `libffi-devel`)
*   `python-dev` (e.g., `python3.10-dev` for Python 3.10)

## Quick Examples

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

bot.run("token")
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

bot.run("token")
```

Explore more examples in the `examples` directory of the repository.

**Note:** Always protect your bot token. Never share it publicly.

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Pycord Guide](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Discord Developers Server](https://discord.gg/discord-developers)
*   [Translations](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)