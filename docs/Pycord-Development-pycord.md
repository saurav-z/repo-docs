# Pycord: The Modern Python Library for Discord Bots

**Pycord** is a powerful and easy-to-use Python library, enabling developers to build feature-rich and asynchronous Discord bots with ease.  [See the original repository on GitHub](https://github.com/Pycord-Development/pycord).

[![PyPI Version](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest Release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord Server](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![Crowdin](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

## Key Features

*   **Asynchronous Programming:** Built with `async` and `await` for modern, non-blocking bot development.
*   **Efficient Rate Limit Handling:**  Automatically handles Discord's rate limits, ensuring your bot runs smoothly.
*   **Optimized Performance:** Designed for speed and memory efficiency.
*   **Full Application API Support:**  Access all of Discord's API features.

## Supported Python Versions

Pycord supports Python versions 3.9 through 3.13.

## Installation

**Prerequisites:** Python 3.9 or higher is required.

To install Pycord, choose one of the following methods:

**1. Basic Installation (No Voice Support):**

```bash
# Linux/macOS
python3 -m pip install -U py-cord

# Windows
py -3 -m pip install -U py-cord
```

**2. Installation with Full Voice Support:**

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[voice]"

# Windows
py -3 -m pip install -U py-cord[voice]
```

**3. Installation for Speedup:**

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
py -3 -m pip install -U py-cord[speed]
```

**4. Development Version Installation:**

*   Clone the repository: `git clone https://github.com/Pycord-Development/pycord`
*   Navigate to the directory: `cd pycord`
*   Install: `python3 -m pip install -U .[voice]`

Or, if you don't want to clone the repository:

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord
# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

### Optional Packages for Optimization

*   `PyNaCl` (for voice support)
*   `aiodns`, `brotlipy`, `cchardet` (for aiohttp speedup)
*   `msgspec` (for JSON speedup)

**Important: Linux Voice Support Dependencies**

Before installing voice support on Linux, ensure you have the following packages installed via your system's package manager (e.g., `apt`, `dnf`):

*   `libffi-dev` (or `libffi-devel` on some systems)
*   `python-dev` (e.g., `python3.10-dev` for Python 3.10)

## Quick Example: Slash Commands

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

You can find more code examples in the `examples` directory.

**Important:** Keep your bot token secure and do not share it.

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Pycord Guide](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)

## Translations

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)