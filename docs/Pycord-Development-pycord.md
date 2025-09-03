[![Pycord Logo](https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png)](https://github.com/Pycord-Development/pycord)

# Pycord: The Modern Python Library for Building Discord Bots

Pycord is a powerful and easy-to-use Python library, perfect for building feature-rich and responsive Discord bots.  [Visit the Pycord Repository on GitHub](https://github.com/Pycord-Development/pycord)

## Key Features

*   **Modern and Pythonic:** Built with `async` and `await` for efficient asynchronous operations.
*   **Robust Rate Limit Handling:**  Handles Discord's rate limits automatically, ensuring your bot runs smoothly.
*   **Optimized Performance:**  Designed for both speed and memory efficiency.
*   **Full Application API Support:**  Leverage the complete Discord API for advanced bot functionality.

## Installation

**Requires Python 3.9 - 3.13**

Install the core library:

```bash
# Linux/macOS
python3 -m pip install -U py-cord
# Windows
py -3 -m pip install -U py-cord
```

Install with Voice Support:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[voice]"
# Windows
py -3 -m pip install -U py-cord[voice]
```

Install with speedup packages:

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
py -3 -m pip install -U py-cord[speed]
```

To install the development version:

```bash
# Clone the repository (Optional)
git clone https://github.com/Pycord-Development/pycord
cd pycord
python3 -m pip install -U .[voice]

# or without cloning:
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord
# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

### Optional Packages

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`__, `brotlipy <https://pypi.org/project/brotlipy/>`__, `cchardet <https://pypi.org/project/cchardet/>`__ (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>`__ (for json speedup)

**Important for Linux Voice Support:**  Before installing Pycord with voice support on Linux, ensure you have the following packages installed via your system's package manager:

*   `libffi-dev` (or `libffi-devel`)
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

**Note:**  Keep your bot token secure and never share it publicly.

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Pycord Guide](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)

## Translations

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

---
_This README has been optimized for clarity, searchability, and user experience._