<div align="center">
  <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord v3" width="200">
  <h1>Pycord: The Modern Python Library for Discord Bots</h1>
</div>

**Pycord is a powerful and user-friendly Python library, enabling you to effortlessly create feature-rich and asynchronous Discord bots.** This allows you to create interactive and engaging experiences on the Discord platform.  [Check out the original repository on GitHub!](https://github.com/Pycord-Development/pycord)

<p align="center">
  <a href="https://pypi.python.org/pypi/py-cord">
    <img src="https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white" alt="PyPI version info">
  </a>
  <a href="https://pypi.python.org/pypi/py-cord">
    <img src="https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white" alt="PyPI supported Python versions">
  </a>
  <a href="https://pypi.python.org/pypi/py-cord">
    <img src="https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge" alt="PyPI downloads">
  </a>
  <a href="https://github.com/Pycord-Development/pycord/releases">
    <img src="https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white" alt="Latest release">
  </a>
  <a href="https://pycord.dev/discord">
    <img src="https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white" alt="Discord server invite">
  </a>
  <a href="https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on">
    <img src="https://badges.crowdin.net/badge/dark/crowdin-on-light.png" alt="Crowdin | Agile localization for tech companies">
  </a>
</p>

## Key Features

*   **Modern Pythonic API:** Leverages `async` and `await` for efficient, non-blocking operations.
*   **Robust Rate Limit Handling:** Automatically manages Discord's rate limits, preventing errors.
*   **Optimized Performance:** Designed for speed and efficient memory usage.
*   **Full Application API Support:** Provides access to all Discord API features.
*   **Easy to Use:** Simplifies bot development with a clean and intuitive API.
*   **Voice Support:**  Integrates voice features using PyNaCl.

## Supported Python Versions

Pycord supports Python versions 3.9 through 3.13.

## Installation

**Prerequisites:** Python 3.9 or higher is required.

To install the library without full voice support:

```bash
# Linux/macOS
python3 -m pip install -U py-cord

# Windows
py -3 -m pip install -U py-cord
```

To install with full voice support:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[voice]"

# Windows
py -3 -m pip install -U py-cord[voice]
```

For additional speedup, install optional packages:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
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

### Optional Packages

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`__, `brotlipy <https://pypi.org/project/brotlipy/>`__, `cchardet <https://pypi.org/project/cchardet/>`__ (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>`__ (for json speedup)

**Important for Linux voice support:**  Before installing, ensure you have these packages installed via your system's package manager (e.g., `apt`, `dnf`):

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

### Traditional Commands Example

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

You can find more code examples in the ``examples`` directory.

**Important:**  Never share your bot token.  Keep it secure!

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Learn how to create Discord bots with Pycord](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)

## Translations

<img src="https://badges.awesome-crowdin.com/translation-200034237-5.png" alt="Translation Status">