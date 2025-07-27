# Xonsh: The Python-Powered Shell 🐍

**Xonsh** is a powerful, cross-platform shell that blends the flexibility of Python with the power of the command line, offering a modern take on shell scripting. [Learn more on GitHub](https://github.com/xonsh/xonsh).

## Key Features

*   **Python-Based:** Xonsh is built on Python 3.6+, making it a superset of Python with added shell primitives.
*   **Cross-Platform:** Works seamlessly across different operating systems.
*   **Shell & Python in One:** Seamlessly execute shell commands and Python code within the same environment.
*   **Extensible with Xontribs:** Extend functionality with a plugin system, including custom commands and integrations.

## How Xonsh Works

Xonsh merges the best of both worlds:

*   **Shell Functionality:** Use familiar shell commands like `cd`, `ls`, `cat`, and custom aliases.
*   **Python Power:** Leverage Python's syntax, libraries, and features directly in your shell environment.

**Example: Shell and Python in action:**

```shell
cd $HOME
id $(whoami)
cat /etc/passwd | grep root > ~/root.txt
$PROMPT = '@ '
```

```python
2 + 2
var = "hello".upper()
import json; json.loads('{"a":1}')
[i for i in range(0,10)]
```
You can use shell commands in Python like this:

```python
len($(curl -L https://xon.sh))
$PATH.append('/tmp')
p'/etc/passwd'.read_text().find('root')
xontrib load dalias
id = $(@json docker ps --format json)['ID']
```
And use python in the shell:
```python
name = 'foo' + 'bar'.upper()
echo @(name) > /tmp/@(name)
ls @(input('file: '))
touch @([f"file{i}" for i in range(0,10)])
aliases['e'] = 'echo @(2+2)'
aliases['a'] = lambda args: print(args)
```

## Getting Started

1.  **Installation:**

    ```shell
    python -m pip install 'xonsh[full]'
    ```

2.  **Resources:**

    *   [Installation Guide](https://xon.sh/contents.html#installation)
    *   [Tutorial](https://xon.sh/tutorial.html)

## Extensions

Xonsh uses plugins called `xontribs` to add functionality.

*   [Xontribs on GitHub](https://github.com/topics/xontrib)
*   [Awesome xontribs](https://github.com/xonsh/awesome-xontribs)
*   [Core xontribs](https://xon.sh/api/_autosummary/xontribs/xontrib.html)
*   [Create a xontrib step by step from template](https://github.com/xonsh/xontrib-template)

## Projects using Xonsh

Xonsh integrates well with several existing projects:

*   conda and mamba
*   Starship
*   zoxide
*   gitsome
*   xxh
*   Snakemake
*   any-nix-shell
*   lix
*   x-cmd
*   rever
*   Regro autotick bot
*   Jupyter and JupyterLab
*   euporie
*   Jupytext

## Join the Xonsh Community

Contribute to the Xonsh community by:

*   Helping with [issues](https://github.com/xonsh/xonsh/issues)
*   Creating xontribs
*   Contributing to the API
*   Writing about xonsh
*   Sponsoring xonsh on GitHub
*   Giving the repository a star

We welcome all contributions!

## Credits

*   Thanks to [Zulip](https://zulip.com/) for supporting the Xonsh community!