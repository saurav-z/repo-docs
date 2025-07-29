# Xonsh: A Python-Powered Shell for a Smarter Command Line

**Xonsh** is a powerful, cross-platform shell that combines the flexibility of Python with the efficiency of a command-line interface.  Get started today and explore the possibilities!  (See the [original repo](https://github.com/xonsh/xonsh) for more details.)

<img src="https://avatars.githubusercontent.com/u/17418188?s=200&v=4" alt="Xonsh shell icon." align="left" width="100px">

<br clear="left"/>

## Key Features:

*   **Python-Powered Shell:** Seamlessly integrate Python code and shell commands.
*   **Cross-Platform:** Works on Windows, macOS, and Linux.
*   **Extensible with Xontribs:**  Enhance functionality with a plugin system.
*   **Python Superset:**  Use Python 3.6+ syntax with added shell primitives.
*   **Shell & Python Synergy:** Run shell commands within Python and Python code within the shell.

### Shell vs. Python:

| Xonsh is the Shell | Xonsh is Python |
| ------------------ | --------------- |
| `cd $HOME`           | `2 + 2`          |
| `id $(whoami)`      | `var = "hello".upper()`|
| `cat /etc/passwd | grep root > ~/root.txt` | `import json; json.loads('{"a":1}')`|
| `$PROMPT = '@ '`    | `[i for i in range(0,10)]` |

### Python in the Shell vs. Shell in Python:

| Xonsh is the Shell in Python | Xonsh is Python in the Shell |
| --------------------------- | -------------------------- |
| `len($(curl -L https://xon.sh))`     | `name = 'foo' + 'bar'.upper()`|
| `$PATH.append('/tmp')`          | `echo @(name) > /tmp/@(name)` |
| `p'/etc/passwd'.read_text().find('root')` | `ls @(input('file: '))` |
| `xontrib load dalias; id = $(@json docker ps --format json)['ID']` | `touch @([f"file{i}" for i in range(0,10)])` |
|  | `aliases['e'] = 'echo @(2+2)'`|
|  | `aliases['a'] = lambda args: print(args)` |

## Getting Started

### Installation:

Install xonsh using pip:

```bash
python -m pip install 'xonsh[full]'
```

### Resources:

*   **Website:** [https://xon.sh](https://xon.sh)
    *   [Installation Guide](https://xon.sh/contents.html#installation)
    *   [Tutorial](https://xon.sh/tutorial.html)

## Extend Xonsh

Xonsh is highly extensible through its xontribs (extensions/plugins) system.

*   **Xontribs on Github:** [https://github.com/topics/xontrib](https://github.com/topics/xontrib)
*   **Awesome xontribs:** [https://github.com/xonsh/awesome-xontribs](https://github.com/xonsh/awesome-xontribs)

## Projects Using Xonsh

Xonsh is compatible with various tools and projects:

*   conda, mamba
*   Starship
*   zoxide
*   gitsome
*   xxh
*   Snakemake
*   any-nix-shell, lix
*   x-cmd
*   rever
*   Regro autotick bot
*   Jupyter, JupyterLab, euporie, and Jupytext (via `xontrib-jupyter`)

## Join the Xonsh Community

*   **Contribute:**  Help improve xonsh by solving issues, creating xontribs, contributing to the API, and more! See the [Developer guide](https://xon.sh/devguide.html).
*   **Sponsor:** Support the project on [GitHub Sponsors](https://github.com/sponsors/xonsh).
*   **Share:** Spread the word about xonsh.

*   **Zulip Community:** [https://xonsh.zulipchat.com/](https://xonsh.zulipchat.com/)

[![Zulip Community](https://img.shields.io/badge/Zulip%20Community-xonsh-green)](https://xonsh.zulipchat.com/)
[![GitHub Actions](https://github.com/xonsh/xonsh/actions/workflows/test.yml/badge.svg)](https://github.com/xonsh/xonsh/actions/workflows/test.yml)
[![codecov.io](https://codecov.io/gh/xonsh/xonsh/branch/master/graphs/badge.svg?branch=main)](https://codecov.io/github/xonsh/xonsh?branch=main)
[![repology.org](https://repology.org/badge/tiny-repos/xonsh.svg)](https://repology.org/project/xonsh/versions)