# Xonsh: The Python-Powered Shell

**Xonsh is a powerful, cross-platform shell that combines the flexibility of Python with the functionality of a command-line interface.** Check out the [official Xonsh repository](https://github.com/xonsh/xonsh).

## Key Features:

*   **Python Integration:** Seamlessly blend Python code and shell commands.
*   **Cross-Platform:** Works on Linux, macOS, and Windows.
*   **Extensible:** Supports a plugin system ("xontribs") for customization.
*   **Shell Primitives:** Includes standard shell functionalities.
*   **Modern:** Python 3.6+ compatible.

## Xonsh: Shell + Python

Xonsh lets you use shell commands and Python code in the same environment.

*   **Shell Commands:**

    ```shell
    cd $HOME
    id $(whoami)
    cat /etc/passwd | grep root > ~/root.txt
    $PROMPT = '@ '
    ```

*   **Python Code:**

    ```python
    2 + 2
    var = "hello".upper()
    import json; json.loads('{"a":1}')
    [i for i in range(0,10)]
    ```

*   **Python in the Shell:**

    ```python
    len($(curl -L https://xon.sh))
    $PATH.append('/tmp')
    p'/etc/passwd'.read_text().find('root')
    xontrib load dalias
    id = $(@json docker ps --format json)['ID']
    ```

*   **Shell in Python:**

    ```python
    name = 'foo' + 'bar'.upper()
    echo @(name) > /tmp/@(name)
    ls @(input('file: '))
    touch @([f"file{i}" for i in range(0,10)])
    aliases['e'] = 'echo @(2+2)'
    aliases['a'] = lambda args: print(args)
    ```

## Getting Started

**Install Xonsh:**

```shell
python -m pip install 'xonsh[full]'
```

**Learn more:**

*   [Installation](https://xon.sh/contents.html#installation)
*   [Tutorial](https://xon.sh/tutorial.html)

## Extensions (Xontribs)

Extend Xonsh's functionality with plugins.

*   [Xontribs on Github](https://github.com/topics/xontrib)
*   [Awesome xontribs](https://github.com/xonsh/awesome-xontribs)
*   [Core xontribs](https://xon.sh/api/_autosummary/xontribs/xontrib.html)
*   [Create a xontrib from template](https://github.com/xonsh/xontrib-template)

## Projects Using Xonsh

Xonsh integrates with various tools:

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

## Jupyter Integration

*   [xontrib-jupyter](https://github.com/xonsh/xontrib-jupyter)
*   Jupyter and JupyterLab
*   euporie
*   Jupytext

## Community

Contribute to Xonsh! Ways to get involved:

*   Help with [issues](https://github.com/xonsh/xonsh/issues)
*   Create a new xontrib.
*   Contribute to the API.
*   Improve documentation and website.
*   Become a sponsor.
*   Spread the word!

## Credits

*   Thanks to [Zulip](https://zulip.com/) for supporting the community.