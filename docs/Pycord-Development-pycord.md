<div align="center">
  <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord v3" width="200">
  <h1>Pycord: The Modern Python Library for Discord Bots</h1>
</div>

Pycord is a powerful and easy-to-use Python library for building feature-rich and asynchronous Discord bots.  [Check out the original repository](https://github.com/Pycord-Development/pycord) for more details.

<p align="center">
  <a href="https://pypi.python.org/pypi/py-cord">
    <img src="https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white" alt="PyPI version">
  </a>
  <a href="https://pypi.python.org/pypi/py-cord">
    <img src="https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python versions">
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
  <a href="https://github.com/sponsors/Pycord-Development">
    <img src="https://img.shields.io/github/sponsors/Pycord-Development?style=for-the-badge" alt="GitHub Sponsors">
  </a>
  <a href="https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on">
    <img src="https://badges.crowdin.net/badge/dark/crowdin-on-light.png" alt="Crowdin">
  </a>
</p>

## Key Features

*   **Modern Pythonic API:** Leverages `async` and `await` for efficient and responsive bot development.
*   **Robust Rate Limit Handling:** Automatically manages Discord's rate limits to ensure smooth operation.
*   **Optimized Performance:** Designed for both speed and efficient memory usage.
*   **Full Application API Support:** Access the complete Discord API for comprehensive bot functionality.

## Supported Python Versions

Pycord supports Python versions **3.9** through **3.13**.

## Installation

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

### Installation for Speedup

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
py -3 -m pip install -U py-cord[speed]
```

### Development Version

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

### Optional Packages for Speedup

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`__, `brotlipy <https://pypi.org/project/brotlipy/>`__, `cchardet <https://pypi.org/project/cchardet/>`__ (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>`__ (for json speedup)

**Note for Linux Voice Support:**  Ensure you have the following packages installed via your system's package manager (e.g., `apt`, `dnf`) *before* installing Pycord with voice support:

*   `libffi-dev` (or `libffi-devel` on some systems)
*   `python-dev` (e.g., `python3.10-dev` for Python 3.10)

## Quick Start Example

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

Find more examples in the `examples` directory of the repository.

**Important:**  Never share your bot token with anyone. It grants full access to your bot.

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Pycord Guide: Learn how to create Discord bots](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)

## Translations

<a href="https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on">
    <img src="https://badges.awesome-crowdin.com/translation-200034237-5.png" alt="Translation Status">
</a>