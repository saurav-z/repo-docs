# Pycord: A Modern Python Library for Discord Bots

**Pycord** is a powerful and easy-to-use Python library, offering a robust and asynchronous API wrapper for building feature-rich Discord bots.  [View the original repository](https://github.com/Pycord-Development/pycord).

[![PyPI Version](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest Release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord Server](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![Crowdin](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

## Key Features

*   **Asynchronous and Modern:**  Built with `async` and `await` for a smooth and responsive bot experience.
*   **Efficient Rate Limit Handling:** Automatically manages Discord's rate limits to keep your bot running smoothly.
*   **Optimized Performance:** Designed for both speed and minimal memory usage.
*   **Full Application API Support:** Leverage the complete Discord API for extensive bot functionality.

## Supported Python Versions

Pycord supports Python versions 3.9 through 3.13.

## Installation

### Prerequisites

*   Python 3.9 or higher

### Installation Options

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

**Install for speedup (additional packages):**

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
py -3 -m pip install -U py-cord[speed]
```

**Install Development Version:**

```bash
$ git clone https://github.com/Pycord-Development/pycord
$ cd pycord
$ python3 -m pip install -U .[voice]
```

**OR**

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

**Important for Linux Voice Support:**  Before installing voice support on Linux, make sure you install the following packages through your system's package manager (e.g., `apt`, `dnf`):

*   libffi-dev (or `libffi-devel` on some systems)
*   python-dev (e.g., `python3.10-dev` for Python 3.10)

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

Find more code examples in the `examples` directory.

**Important Note:** Never share your bot token with anyone. It grants access to your bot.

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Pycord Guide (Learn to Create Bots)](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)

## Translations

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)