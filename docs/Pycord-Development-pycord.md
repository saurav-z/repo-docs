<!-- Improved README for Pycord -->

![Pycord v3](https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png)

# Pycord: The Modern Python Library for Discord Bots

**Pycord is a powerful and user-friendly Python library, making it easy to build feature-rich and asynchronous Discord bots.**

[View the original repository on GitHub](https://github.com/Pycord-Development/pycord)

## Key Features

*   **Modern & Pythonic:** Designed with `async` and `await` for efficient, non-blocking operations.
*   **Robust Rate Limit Handling:** Automatically manages Discord's rate limits to ensure your bot runs smoothly.
*   **Optimized Performance:** Built for speed and efficiency, minimizing resource usage.
*   **Full Application API Support:** Access all of Discord's API features for maximum bot customization.

## Supported Python Versions

Pycord supports Python versions 3.9 through 3.13.

## Installation

**Prerequisites:** Python 3.9 or higher is required.

**Basic Installation:**

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

**Development Version Installation:**

```bash
git clone https://github.com/Pycord-Development/pycord
cd pycord
python3 -m pip install -U .[voice]
```

**Alternatively:**

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

**Important for Linux Voice Support:** Before installing with voice support, ensure you have the following packages installed using your system's package manager (e.g., `apt`, `dnf`):

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

**Note:** Secure your bot token; do not share it with anyone.

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Learn how to create Discord bots with Pycord](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)

## Translations

![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)