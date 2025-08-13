<!--
  SPDX-License-Identifier: CC0-1.0
-->
<p align="center">
  <a href="https://github.com/Pycord-Development/pycord">
    <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord v3" width="200">
  </a>
</p>

# Pycord: The Modern Python Library for Discord Bots

**Pycord is a powerful and user-friendly Python library that makes building feature-rich Discord bots a breeze.**

[![PyPI Version](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![Python Versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest Release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord Server](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![Crowdin](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

## Key Features

*   **Asynchronous and Modern:** Built with `async` and `await` for efficient, non-blocking operations.
*   **Rate Limit Handling:** Automatic and robust rate limit management to keep your bot running smoothly.
*   **Optimized Performance:** Designed for speed and minimal memory usage.
*   **Full Discord API Coverage:** Supports all features of the Discord application API.
*   **Easy to Use:** User-friendly API makes bot development accessible to everyone.

## Installation

**Requires Python 3.9 or higher.**

### Basic Installation

```bash
# Linux/macOS
python3 -m pip install -U py-cord

# Windows
py -3 -m pip install -U py-cord
```

### Installation with Voice Support

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[voice]"

# Windows
py -3 -m pip install -U py-cord[voice]
```

**Note for Linux Voice Support:** You may need to install dependencies like `libffi-dev` and `python3.x-dev` (where x is your Python version) using your system's package manager (e.g., `apt`, `dnf`).

### Installation with Speedup Packages

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
py -3 -m pip install -U py-cord[speed]
```

### Development Version Installation

```bash
$ git clone https://github.com/Pycord-Development/pycord
$ cd pycord
$ python3 -m pip install -U .[voice]
```

Or, without cloning:

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord
# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

### Optional Packages for Speedup

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`, `brotlipy <https://pypi.org/project/brotlipy/>`, `cchardet <https://pypi.org/project/cchardet/>` (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>` (for json speedup)

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

**Important:**  Never share your bot's token!

## Useful Links

*   **Documentation:** [https://docs.pycord.dev/en/master/index.html](https://docs.pycord.dev/en/master/index.html)
*   **Bot Creation Guide:** [https://guide.pycord.dev](https://guide.pycord.dev)
*   **Official Discord Server:** [https://pycord.dev/discord](https://pycord.dev/discord)
*   **Discord Developers Server:** [https://discord.gg/discord-developers](https://discord.gg/discord-developers)

## Translations

<img src="https://badges.awesome-crowdin.com/translation-200034237-5.png" alt="Translation Status">

## Contributing

See the [contributing guide](https://github.com/Pycord-Development/pycord/blob/master/.github/CONTRIBUTING.md) to get started.

**[Back to the Pycord Repository](https://github.com/Pycord-Development/pycord)**