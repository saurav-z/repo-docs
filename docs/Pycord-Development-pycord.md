<!-- Improved README.md for Pycord -->

<!-- Pycord Banner (replace with actual image link) -->
<p align="center">
  <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord v3" width="600">
</p>

<!-- Title and Description -->
# Pycord: The Modern Python Library for Discord Bots

**Pycord is a feature-rich and easy-to-use Python library that simplifies building powerful and engaging Discord bots.**  This library offers a streamlined, asynchronous-first API that lets you create bots for your Discord servers with ease.

<!-- Badges -->
[![PyPI Version](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest Release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord Server](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Pycord-Development?style=for-the-badge)](https://github.com/sponsors/Pycord-Development)
[![Crowdin](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

<!-- Key Features -->
## Key Features of Pycord:

*   **Asynchronous API:** Leverage modern Python's `async` and `await` for efficient, non-blocking bot development.
*   **Rate Limit Handling:** Pycord automatically handles Discord's rate limits, so your bot stays online.
*   **Optimized Performance:** Built for both speed and low memory usage to deliver a responsive bot experience.
*   **Full Application API Support:**  Access and utilize all features of the Discord API, including slash commands and application commands.

<!-- Supported Python Versions -->
## Python Compatibility

Pycord supports Python versions **3.9** through **3.13**.

<!-- Installation Instructions -->
## Installation

To get started with Pycord, follow these installation steps:

**Prerequisites:** Python 3.9 or higher

**Basic Installation (without voice support):**

```bash
# Linux/macOS
python3 -m pip install -U py-cord

# Windows
py -3 -m pip install -U py-cord
```

**Installation with Voice Support:**

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[voice]"

# Windows
py -3 -m pip install -U py-cord[voice]
```

**Installation with Speedup Packages:**
```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
py -3 -m pip install -U py-cord[speed]
```

**Installing the Development Version:**

```bash
git clone https://github.com/Pycord-Development/pycord
cd pycord
python3 -m pip install -U .[voice]
```

or
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

**Note for Linux Voice Support:** Before installing voice support on Linux, ensure you have the following packages installed through your system's package manager (e.g., `apt`, `dnf`):

*   `libffi-dev` (or `libffi-devel` on some systems)
*   `python-dev` (e.g., `python3.10-dev` for Python 3.10)

<!-- Quick Example -->
## Quick Start Example

Here's a simple example to get you started:

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
Find more examples in the `examples` directory of the [Pycord repository](https://github.com/Pycord-Development/pycord).

**Important:**  Never share your bot token.  It grants full access to your bot.

<!-- Useful Links -->
## Useful Resources

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Pycord Guide (Learn how to create Discord bots)](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)
*   [Pycord Repository](https://github.com/Pycord-Development/pycord)

<!-- Translations -->
## Translations

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)