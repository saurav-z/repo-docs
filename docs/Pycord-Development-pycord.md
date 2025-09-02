<!--  Pycord Logo Image -->
<img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord v3" width="200"/>

# Pycord: The Modern Python Library for Discord Bots

**Pycord** is your go-to solution for building powerful and feature-rich Discord bots in Python, offering an intuitive and efficient way to interact with the Discord API.

[![PyPI Version](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest Release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord Server](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Pycord-Development?style=for-the-badge)](https://github.com/sponsors/Pycord-Development)
[![Crowdin Translations](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

## Key Features

*   **Modern Pythonic API:** Leveraging `async` and `await` for efficient and responsive bot development.
*   **Robust Rate Limit Handling:**  Automatically manages Discord's rate limits to ensure your bot runs smoothly.
*   **Optimized Performance:** Designed for both speed and efficient memory usage.
*   **Full Application API Support:** Access to the complete range of Discord API features.

## Installation

**Requires Python 3.9 or higher.**

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

**Install for a performance boost:**

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"

# Windows
py -3 -m pip install -U py-cord[speed]
```

**Install the development version:**

```bash
git clone https://github.com/Pycord-Development/pycord
cd pycord
python3 -m pip install -U .[voice]
```

**Alternative development installation (without cloning):**

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord

# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

### Optional Packages

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`, `brotlipy <https://pypi.org/project/brotlipy/>`, `cchardet <https://pypi.org/project/cchardet/>` (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>` (for json speedup)

**Important for Linux Voice Support:**  Before installing with voice support on Linux, ensure you have the following packages installed through your system's package manager (e.g., `apt`, `dnf`):

*   `libffi-dev` (or `libffi-devel` on some systems)
*   `python-dev` (e.g. `python3.10-dev` for Python 3.10)

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

*   More code examples can be found in the ``examples`` directory.
*   **Important Security Note:** Never share your bot token.

## Useful Links

*   [Pycord Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Pycord Guide - Learn How to Create Discord Bots](https://guide.pycord.dev)
*   [Pycord Official Discord Server](https://pycord.dev/discord)
*   [Discord Developers Server](https://discord.gg/discord-developers)

## Translations

<img src="https://badges.awesome-crowdin.com/translation-200034237-5.png" alt="Translation Status">

---

**[View the source code on GitHub](https://github.com/Pycord-Development/pycord)**