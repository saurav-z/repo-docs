# Xonsh: A Python-Powered, Cross-Platform Shell

Xonsh is a powerful, cross-platform shell that combines the best of Python with the flexibility of a command-line interface. ([Original Repo](https://github.com/xonsh/xonsh))

## Key Features

*   **Pythonic Shell:** Xonsh is a superset of Python 3.6+, allowing you to use Python syntax and libraries directly in your shell.
*   **Cross-Platform Compatibility:** Works seamlessly across various operating systems, including Windows, macOS, and Linux.
*   **Shell Primitives:** Includes built-in shell commands and features like command substitution, environment variable access, and more.
*   **Extensible with Xontribs:**  Enhance your shell with a rich ecosystem of plugins called "xontribs."
*   **Integrated with Python:**  Leverage Python's vast ecosystem for scripting, automation, and data analysis directly within your shell.

## Xonsh: Shell Meets Python

Xonsh cleverly merges shell commands and Python code, allowing you to seamlessly switch between them. Here's how:

*   **Shell Commands:** Use familiar commands like `cd`, `ls`, and `cat`.

    ```shell
    cd $HOME
    cat /etc/passwd | grep root > ~/root.txt
    ```

*   **Python Code:** Execute Python statements, import libraries, and define functions.

    ```python
    2 + 2
    var = "hello".upper()
    import json; json.loads('{"a":1}')
    ```

*   **Mix and Match:** Combine shell commands with Python logic and access shell variables.

    ```python
    len($(curl -L https://xon.sh))
    $PATH.append('/tmp')
    p'/etc/passwd'.read_text().find('root')
    ```

## Getting Started

1.  **Installation:** Install Xonsh using pip:

    ```bash
    python -m pip install 'xonsh[full]'
    ```

2.  **Explore:** Visit the Xonsh website for detailed documentation and tutorials:

    *   [Installation](https://xon.sh/contents.html#installation)
    *   [Tutorial](https://xon.sh/tutorial.html)

## Xontribs: Extend Xonsh Functionality

Xonsh's plugin system, "xontribs," allows you to add custom commands and features:

*   [Xontribs on Github](https://github.com/topics/xontrib)
*   [Awesome xontribs](https://github.com/xonsh/awesome-xontribs)
*   [Core xontribs](https://xon.sh/api/_autosummary/xontribs/xontrib.html)

## Projects Leveraging Xonsh

Xonsh seamlessly integrates with various tools and projects:

*   **Package Managers:** conda, mamba
*   **Shell Enhancements:** Starship, zoxide, gitsome, xxh
*   **Workflow Systems:** Snakemake
*   **Nix Integration:** any-nix-shell, lix
*   **Other Utilities:** x-cmd, rever, Regro autotick bot
*   **Jupyter Integration:**  xontrib-jupyter, Jupyter, JupyterLab, euporie, Jupytext

## Join the Xonsh Community

Contribute to Xonsh by:

*   Solving issues.
*   Creating new xontribs.
*   Contributing to the API.
*   Implementing support in third-party tools.
*   Designing logos/images and improving the website.
*   Becoming a sponsor.
*   Sharing Xonsh on social media.
*   Giving the repo and xontribs a star!

## Credits

Thanks to Zulip for supporting the Xonsh community!