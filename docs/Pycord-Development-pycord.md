# Pycord: The Modern Python Library for Discord Bots

**Pycord** is the go-to library for building powerful and feature-rich Discord bots in Python, offering a modern and intuitive API. Find the source code on GitHub: [Pycord-Development/pycord](https://github.com/Pycord-Development/pycord).

## Key Features

*   **Asynchronous Programming:** Leverage the power of `async` and `await` for efficient, non-blocking operations.
*   **Rate Limit Handling:**  Enjoy automatic and robust rate limit management to keep your bot running smoothly.
*   **Optimized Performance:** Experience a library built for speed and efficient memory usage.
*   **Full Application API Support:**  Access the complete Discord API functionality.

## Installation

**Requirements:** Python 3.9 - 3.13

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

**Installation for Speedup (Optional):**

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"

# Windows
py -3 -m pip install -U py-cord[speed]
```

**Installing the Development Version:**

```bash
# Clone the repository (Recommended)
git clone https://github.com/Pycord-Development/pycord
cd pycord
python3 -m pip install -U .[voice]

# Alternatively, install directly from GitHub
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord

# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

### Optional Packages for Enhanced Performance

*   `PyNaCl <https://pypi.org/project/PyNaCl/>` (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`, `brotlipy <https://pypi.org/project/brotlipy/>`, `cchardet <https://pypi.org/project/cchardet/>` (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>` (for JSON speedup)

**Important Note for Linux Voice Support:** Before installing with voice support on Linux, ensure you have installed the following packages via your system's package manager (e.g., `apt`, `dnf`):

*   `libffi-dev` (or `libffi-devel` on some systems)
*   `python-dev` (e.g., `python3.10-dev` for Python 3.10)

## Quick Start Examples

**Slash Commands Example:**

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

**Traditional Commands Example:**

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

Explore more examples in the `examples` directory of the repository.

**Security Note:**  Keep your bot token secure and never share it publicly.

## Useful Links

*   **Documentation:** [https://docs.pycord.dev/en/master/index.html](https://docs.pycord.dev/en/master/index.html)
*   **Getting Started Guide:** [https://guide.pycord.dev](https://guide.pycord.dev)
*   **Official Discord Server:** [https://pycord.dev/discord](https://pycord.dev/discord)
*   **Official Discord Developers Server:** [https://discord.gg/discord-developers](https://discord.gg/discord-developers)

## Translations

[![Translation status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)