<!-- Improved README for Pycord -->

<div align="center">
  <a href="https://github.com/Pycord-Development/pycord">
    <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord v3" width="200">
  </a>
  <h1>Pycord: A Modern Python Discord API Wrapper</h1>
</div>

Pycord is the premier choice for building robust and feature-rich Discord bots in Python, offering an easy-to-use and asynchronous-ready API. ([Original Repository](https://github.com/Pycord-Development/pycord))

[![PyPI Version](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest Release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord Server](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Pycord-Development?style=for-the-badge)](https://github.com/sponsors/Pycord-Development)
[![Crowdin](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

## Key Features

*   **Modern Pythonic API:** Leverage the power of `async` and `await` for efficient bot development.
*   **Robust Rate Limit Handling:** Pycord automatically manages Discord's rate limits, ensuring your bot runs smoothly.
*   **Optimized Performance:** Built for speed and efficiency, minimizing resource usage.
*   **Full Application API Support:** Access the complete Discord API, allowing for advanced bot features.

## Getting Started

### Prerequisites

*   Python 3.9 - 3.13

### Installation

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

**Install Development Version:**

```bash
git clone https://github.com/Pycord-Development/pycord
cd pycord
python3 -m pip install -U .[voice]
```

or without cloning:

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord
# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

### Optional Packages

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`__, `brotlipy <https://pypi.org/project/brotlipy/>`__, `cchardet <https://pypi.org/project/cchardet/>`__ (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>`__ (for json speedup)

**Important for Linux Voice Support:**  Before installing voice support, ensure you have installed:

*   `libffi-dev` (or `libffi-devel`)
*   `python-dev` (e.g., `python3.10-dev`)

## Quick Example (Slash Commands)

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

**Note:** Always protect your bot token.

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Pycord Guide](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Discord Developers Server](https://discord.gg/discord-developers)

## Translations

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)