# Pycord: The Modern Python Library for Discord Bots

**Pycord** is a powerful and user-friendly Python library that empowers you to build feature-rich and efficient Discord bots. Check out the original repository [here](https://github.com/Pycord-Development/pycord)!

## Key Features:

*   **Asynchronous & Modern:** Built with async/await for efficient and responsive bot development.
*   **Rate Limit Handling:** Automatically manages Discord API rate limits, ensuring your bot runs smoothly.
*   **Optimized Performance:** Designed for speed and minimal memory usage.
*   **Full Application API Support:** Access all of Discord's API features.

## Installation

**Requirements:** Python 3.9 or higher is required.

**Basic Installation (no voice support):**

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

**Installing the development version:**

```bash
$ git clone https://github.com/Pycord-Development/pycord
$ cd pycord
$ python3 -m pip install -U .[voice]
```

**OR, without cloning:**

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord

# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

**Optional Packages for Speedup:**

*   `PyNaCl`: For voice support.  Requires `libffi-dev` and `python-dev` on Linux.
*   `aiodns`, `brotlipy`, `cchardet`: For aiohttp speedup.
*   `msgspec`: For JSON speedup.

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

*   **Important:** Never share your bot token!

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Guide](https://guide.pycord.dev)
*   [Discord Server](https://pycord.dev/discord)
*   [Discord Developers Server](https://discord.gg/discord-developers)

## Translations

[Translation Status](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)