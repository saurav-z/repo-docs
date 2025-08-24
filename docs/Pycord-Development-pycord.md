<!-- Improved README for Pycord - A Python Discord API Wrapper -->

<div align="center">
  <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord v3" width="200">
  <h1>Pycord: The Modern Python Library for Discord Bots</h1>
  <p><em>Build powerful and feature-rich Discord bots with ease using Pycord, a cutting-edge Python API wrapper.</em></p>
  <a href="https://github.com/Pycord-Development/pycord">
    <img src="https://img.shields.io/github/stars/Pycord-Development/pycord?style=social" alt="GitHub Stars">
  </a>
</div>

---

## About Pycord

Pycord is a robust, user-friendly, and asynchronous-ready Python library designed to simplify the creation of Discord bots.  It offers a comprehensive set of features and an intuitive API, making it the ideal choice for both novice and experienced developers.

**[View the Pycord GitHub Repository](https://github.com/Pycord-Development/pycord)**

## Key Features

*   **Modern & Pythonic:** Built with `async` and `await` for efficient asynchronous operations.
*   **Robust Rate Limit Handling:**  Automatically manages Discord's rate limits to ensure smooth bot operation.
*   **Optimized Performance:** Designed for both speed and efficient memory usage.
*   **Full API Coverage:** Provides support for the complete Discord application API.
*   **User-Friendly:** Easy to learn and use with comprehensive documentation and examples.

## Installation

**Requirements:** Python 3.9 or higher is required.

**Basic Installation (No Voice Support):**

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

**Installation for speedup:**

```bash
# Linux/macOS
python3 -m pip install -U "py-cord[speed]"
# Windows
py -3 -m pip install -U py-cord[speed]
```

**Installing the Development Version:**

```bash
$ git clone https://github.com/Pycord-Development/pycord
$ cd pycord
$ python3 -m pip install -U .[voice]
```

or:

```bash
# Linux/macOS
python3 -m pip install git+https://github.com/Pycord-Development/pycord
# Windows
py -3 -m pip install git+https://github.com/Pycord-Development/pycord
```

**Optional Packages**

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`__, `brotlipy <https://pypi.org/project/brotlipy/>`__, `cchardet <https://pypi.org/project/cchardet/>`__ (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>`__ (for json speedup)

**Linux Voice Support Dependencies:** Before installing with voice support on Linux, ensure you have these packages installed:

*   libffi-dev (or libffi-devel on some systems)
*   python-dev (e.g., python3.10-dev for Python 3.10)

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

bot.run("YOUR_BOT_TOKEN")  # Replace with your bot token
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

bot.run("YOUR_BOT_TOKEN")  # Replace with your bot token
```

**Important:** Keep your bot token confidential; do not share it.

## Useful Links

*   [Pycord Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Pycord Guide: Learn How to Create Discord Bots](https://guide.pycord.dev)
*   [Official Pycord Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)

## Translations

<img src="https://badges.awesome-crowdin.com/translation-200034237-5.png" alt="Translation Status">
```

Key improvements and changes:

*   **SEO Optimization:** Added a clear title, keywords (Python, Discord bot, API wrapper), and a concise description.
*   **Summarization:**  Combined information for brevity and clarity.
*   **Clear Headings:** Structured the README with logical headings for easy navigation.
*   **Bulleted Key Features:** Highlighted the essential features for quick understanding.
*   **One-Sentence Hook:**  Immediately grabs the reader's attention.
*   **Concise Installation Instructions:** Simplified installation steps.
*   **Clear Examples:** Provided basic examples with clear instructions on how to run them.
*   **Direct Links:**  Included direct links to key resources.
*   **GitHub Star Badge:** Added a badge to increase visibility.
*   **Translation Status:** Added translation status badge for internationalization.
*   **Emphasis on Key Aspects:**  The most important aspects (async, rate limits, ease of use) are emphasized.
*   **Removed unnecessary repetition.**
*   **Corrected code formatting.**