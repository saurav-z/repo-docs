<div align="center">
  <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord v3" width="200">
  <h1>Pycord: The Ultimate Python Library for Discord Bots</h1>
</div>

<p align="center">
  <a href="https://pypi.org/project/py-cord">
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
  <a href="https://github.com/sponsors/Pycord-Development">
    <img src="https://img.shields.io/github/sponsors/Pycord-Development?style=for-the-badge" alt="GitHub Sponsors">
  </a>
  <a href="https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on">
    <img src="https://badges.crowdin.net/badge/dark/crowdin-on-light.png" alt="Crowdin | Agile localization for tech companies">
  </a>
</p>

**Pycord is the go-to Python library for building powerful, modern, and feature-rich Discord bots.** Designed with ease of use and performance in mind, Pycord empowers developers to create engaging experiences on the Discord platform.

## Key Features:

*   **Asynchronous and Modern:** Built on `async` and `await` for a clean, efficient, and modern Pythonic API.
*   **Rate Limit Handling:** Automatic and intelligent rate limit handling to ensure your bot runs smoothly.
*   **Optimized Performance:** Engineered for speed and efficient memory usage.
*   **Full Application API Support:**  Complete access to the Discord API, including slash commands, user commands, and more.

## Installation

**Requires Python 3.9 or higher.**

Install without voice support:

```bash
# Linux/macOS
python3 -m pip install -U py-cord

# Windows
py -3 -m pip install -U py-cord
```

Install with full voice support:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[voice]"

# Windows
py -3 -m pip install -U py-cord[voice]
```

Install additional packages for speedup:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
py -3 -m pip install -U py-cord[speed]
```

Install the development version:

```bash
git clone https://github.com/Pycord-Development/pycord
cd pycord
python3 -m pip install -U .[voice]
```

Or without cloning:

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord
# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

### Optional Packages

*   `PyNaCl <https://pypi.org/project/PyNaCl/>` (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`, `brotlipy <https://pypi.org/project/brotlipy/>`, `cchardet <https://pypi.org/project/cchardet/>` (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>` (for json speedup)

**Important for Linux voice support:** Ensure you have the necessary development packages installed (e.g., `libffi-dev` or `libffi-devel`, and `python3.x-dev`) before installing `py-cord[voice]`.

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

Find more examples in the `examples` directory.  **Never share your bot token.**

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Bot Creation Guide](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)
*   [GitHub Repository](https://github.com/Pycord-Development/pycord)

## Translations

<img src="https://badges.awesome-crowdin.com/translation-200034237-5.png" alt="Translation Status">