<div align="center">
  <a href="https://github.com/Pycord-Development/pycord">
    <img src="https://raw.githubusercontent.com/Pycord-Development/pycord/master/pycord.png" alt="Pycord Logo" width="200"/>
  </a>
  <br>
  <a href="https://github.com/Pycord-Development/pycord">Pycord</a> - The modern, feature-rich, and easy-to-use Python library for building powerful Discord bots.
</div>

---

## Pycord: Your Gateway to Building Advanced Discord Bots

Pycord is a robust and actively maintained Python library designed to simplify the creation of Discord bots.  It offers a comprehensive set of features, making it an ideal choice for both beginners and experienced developers.

### Key Features:

*   **Modern Pythonic API:** Leverages `async` and `await` for efficient, non-blocking operations.
*   **Robust Rate Limit Handling:**  Automatically manages Discord's rate limits, ensuring your bot runs smoothly.
*   **Optimized Performance:** Designed for both speed and minimal memory usage.
*   **Full Application API Support:**  Provides complete access to Discord's application features.
*   **Active Development and Community Support:**  Benefit from a constantly evolving library, and a helpful community.

[![PyPI version](https://img.shields.io/pypi/v/py-cord.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![Python versions](https://img.shields.io/pypi/pyversions/py-cord.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/py-cord)
[![PyPI downloads](https://img.shields.io/pypi/dm/py-cord?color=blueviolet&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.python.org/pypi/py-cord)
[![Latest Release](https://img.shields.io/github/v/release/Pycord-Development/pycord?include_prereleases&label=Latest%20Release&logo=github&sort=semver&style=for-the-badge&logoColor=white)](https://github.com/Pycord-Development/pycord/releases)
[![Discord server invite](https://img.shields.io/discord/881207955029110855?label=discord&style=for-the-badge&logo=discord&color=5865F2&logoColor=white)](https://pycord.dev/discord)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Pycord-Development?style=for-the-badge)](https://github.com/sponsors/Pycord-Development)
[![Crowdin](https://badges.crowdin.net/badge/dark/crowdin-on-light.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)

---

**Supported Python Versions:**  Pycord supports Python 3.9 - 3.13.

### Installing Pycord

**Prerequisites:** Python 3.9 or higher is required.

**Installation Options:**

*   **Without Voice Support:**
    ```bash
    # Linux/macOS
    python3 -m pip install -U py-cord

    # Windows
    py -3 -m pip install -U py-cord
    ```

*   **With Full Voice Support:**
    ```bash
    # Linux/macOS
    python3 -m pip install -U "py-cord[voice]"

    # Windows
    py -3 -m pip install -U py-cord[voice]
    ```

*   **For Speedup:**
    ```bash
    # Linux/macOS
    python3 -m pip install -U "py-cord[speed]"
    # Windows
    py -3 -m pip install -U py-cord[speed]
    ```

*   **Development Version:**
    ```bash
    $ git clone https://github.com/Pycord-Development/pycord
    $ cd pycord
    $ python3 -m pip install -U .[voice]
    ```

    Or:

    ```bash
    # Linux/macOS
    python3 -m pip install git+https://github.com/Pycord-Development/pycord
    # Windows
    py -3 -m pip install git+https://github.com/Pycord-Development/pycord
    ```

### Optional Packages

*   `PyNaCl <https://pypi.org/project/PyNaCl/>`__ (for voice support)
*   `aiodns <https://pypi.org/project/aiodns/>`, `brotlipy <https://pypi.org/project/brotlipy/>`, `cchardet <https://pypi.org/project/cchardet/>` (for aiohttp speedup)
*   `msgspec <https://pypi.org/project/msgspec/>` (for json speedup)

**Important for Linux voice support:**  Before installing with voice support on Linux, ensure you have the following packages installed via your system's package manager (e.g., `apt`, `dnf`):
*   `libffi-dev` (or `libffi-devel` on some systems)
*   `python-dev` (e.g. `python3.10-dev` for Python 3.10)

### Quickstart Example

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

You can find more code examples in the ``examples`` directory of the [Pycord GitHub Repository](https://github.com/Pycord-Development/pycord).

**Important:** Never share your bot token.  It provides full access to your bot.

---

### Useful Links

*   **Documentation:** [https://docs.pycord.dev/en/master/index.html](https://docs.pycord.dev/en/master/index.html)
*   **Pycord Guide:** [https://guide.pycord.dev](https://guide.pycord.dev)
*   **Official Discord Server:** [https://pycord.dev/discord](https://pycord.dev/discord)
*   **Official Discord Developers Server:** [https://discord.gg/discord-developers](https://discord.gg/discord-developers)
---

### Translations

[![Translation Status](https://badges.awesome-crowdin.com/translation-200034237-5.png)](https://translations.pycord.dev/documentation/?utm_source=badge&utm_medium=referral&utm_campaign=badge-add-on)