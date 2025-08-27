<div align="center">
  <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord Logo" width="200"/>
  <h1>Pycord: The Ultimate Python Library for Discord Bots</h1>
</div>

Pycord is a powerful, easy-to-use, and feature-rich asynchronous Python library for interacting with the Discord API.  ([View on GitHub](https://github.com/Pycord-Development/pycord))

[![PyPI Version](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI Downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest Release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord Server](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![Crowdin](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

**Key Features:**

*   **Modern Pythonic API:** Built with `async` and `await` for efficient, non-blocking operations.
*   **Robust Rate Limit Handling:** Automatically manages Discord's rate limits, so you don't have to.
*   **Optimized Performance:** Designed for speed and low memory usage.
*   **Full Application API Support:** Access all of Discord's features through a comprehensive API.
*   **Slash Command & Context Menu Support**: Leverage Discord's interactive features.

**Requirements:**

*   Python 3.9 - 3.13

**Installation:**

Install the library using pip:

```bash
# For basic installation (no voice support)
python3 -m pip install -U py-cord

# For full voice support
python3 -m pip install -U "py-cord[voice]"

# To install packages for speedup (aiohttp speedup with aiodns, brotli and cchardet, json speedup with msgspec)
python3 -m pip install -U "py-cord[speed]"
```
**Note for Linux Voice Support:** Before installing `py-cord[voice]`, ensure you have `libffi-dev` (or `libffi-devel`) and your Python development headers (e.g., `python3.10-dev` for Python 3.10) installed via your system's package manager (e.g., `apt`, `dnf`).

**Quick Start Example:**

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

bot.run("YOUR_BOT_TOKEN")
```

**Important:**  Never share your bot token.

**Optional Packages for Speedups:**
*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`__, `brotlipy <https://pypi.org/project/brotlipy/>`__, `cchardet <https://pypi.org/project/cchardet/>`__ (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>`__ (for json speedup)


**Useful Links:**

*   [Documentation](https://docs.pycord.dev/en/master/index.html)
*   [Pycord Guide (Learn to create Discord bots)](https://guide.pycord.dev)
*   [Official Discord Server](https://pycord.dev/discord)
*   [Official Discord Developers Server](https://discord.gg/discord-developers)

**Translations:**

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)