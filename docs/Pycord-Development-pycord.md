# Pycord: The Modern Python Library for Discord Bots

**Pycord is a powerful and easy-to-use Python library that allows you to effortlessly create feature-rich Discord bots.** You can find the original repository [here](https://github.com/Pycord-Development/pycord).

## Key Features

*   **Asynchronous API:** Built with `async` and `await` for efficient, non-blocking operations.
*   **Rate Limit Handling:** Automatic and intelligent handling of Discord's rate limits.
*   **Optimized Performance:** Designed for both speed and efficient memory usage.
*   **Full Application API Support:** Comprehensive coverage of the Discord API.

## Installation

**Requires Python 3.9 or higher**

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

### Installation with Speedup Packages

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

Or, without cloning:

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord
# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

## Optional Packages

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`__, `brotlipy <https://pypi.org/project/brotlipy/>`__, `cchardet <https://pypi.org/project/cchardet/>`__ (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>`__ (for json speedup)

**Important for Linux Voice Support:** Before installing with voice support, ensure you have the following packages installed via your system's package manager:

*   `libffi-dev` (or `libffi-devel`)
*   `python-dev` (e.g., `python3.10-dev` for Python 3.10)

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

## Useful Links

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Pycord Guide](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)

## Translations