<div align="center">
  <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord v3" width="200">
  <h1>Pycord: The Modern Python Library for Discord Bots</h1>
</div>

Pycord is a powerful and easy-to-use Python library, built for creating feature-rich Discord bots with a modern and asynchronous approach.  [Explore the original repository](https://github.com/Pycord-Development/pycord) for more information.

<br>

[![PyPI Version](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest Release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord Server](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![Crowdin](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

## Key Features

*   **Modern Pythonic API:** Leverages `async` and `await` for efficient and responsive bot development.
*   **Robust Rate Limit Handling:** Automatically manages Discord's rate limits to ensure smooth operation.
*   **Optimized Performance:** Designed for speed and low memory usage, providing a performant bot experience.
*   **Full Application API Support:**  Access to the complete Discord API for comprehensive bot capabilities.

## Installation

**Prerequisites:** Python 3.9 - 3.13

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

**Installation with Optional Speedup Packages:**

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
py -3 -m pip install -U py-cord[speed]
```

**Development Version Installation:**

```bash
git clone https://github.com/Pycord-Development/pycord
cd pycord
python3 -m pip install -U .[voice]
```

**Or using pip:**

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord
# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

### Optional Packages

*   `PyNaCl`:  (for voice support) - *Required for voice functionality.*
*   `aiodns`, `brotlipy`, `cchardet`:  (for aiohttp speedup) - *Improve network performance.*
*   `msgspec`:  (for JSON speedup) - *Enhance JSON parsing efficiency.*

**Important for Linux Voice Support:** Before installing with voice support, ensure you have the following packages installed via your system's package manager (e.g., `apt`, `dnf`):

*   `libffi-dev` (or `libffi-devel` on some systems)
*   `python-dev` (e.g., `python3.10-dev` for Python 3.10)

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

For more detailed examples, check the `examples` directory in the repository.

**Security Note:** Protect your bot token; do not share it publicly.

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Pycord Guide](https://guide.pycord.dev/) - Learn how to create Discord bots.
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)

## Translations

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)