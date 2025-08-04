# Xonsh: A Python-Powered Shell for a Better Command Line Experience

**Xonsh** is a powerful and versatile shell that combines the flexibility of Python with the functionality of a traditional command-line interface, making it a compelling choice for developers and power users alike.  Check out the original repo for more details: [https://github.com/xonsh/xonsh](https://github.com/xonsh/xonsh).

## Key Features

*   **Python Superpowers:** Xonsh is a superset of Python 3.6+, allowing you to leverage the full power of Python within your shell commands.
*   **Shell Primitives:**  Seamlessly integrates shell commands and Python code, enabling you to execute shell commands, manipulate files, and interact with the operating system using familiar Python syntax.
*   **Cross-Platform Compatibility:** Works seamlessly across various operating systems, including Linux, macOS, and Windows.
*   **Extensible with Xontribs:**  Customize your shell with a rich ecosystem of extensions (xontribs) that add new features and integrations.
*   **Interactive and Scriptable:**  Use Xonsh interactively for everyday tasks or write powerful scripts to automate complex workflows.

## Xonsh in Action: Shell and Python Combined

Xonsh lets you use the best of both worlds, with the syntax you know:

| **Xonsh Shell**                         | **Python**                                  |
| ---------------------------------------- | ------------------------------------------- |
| `cd $HOME`                             | `2 + 2`                                     |
| `cat /etc/passwd | grep root > ~/root.txt` | `var = "hello".upper()`                    |
| `$PROMPT = '@ '`                        | `import json; json.loads('{"a":1}')`         |

## Getting Started

1.  **Install:**

    ```bash
    python -m pip install 'xonsh[full]'
    ```

2.  **Learn More:**

    *   **Installation:** Explore various installation methods on the [official website](https://xon.sh/contents.html#installation).
    *   **Tutorial:** Get started with the [step-by-step tutorial](https://xon.sh/tutorial.html).

## Extensions (Xontribs)

Enhance your Xonsh experience with the extension/plugin system:

*   **Explore Xontribs:** Find a wide variety of extensions on [GitHub](https://github.com/topics/xontrib) and the [Awesome xontribs](https://github.com/xonsh/awesome-xontribs).
*   **Create Your Own:** Get started with the [xontrib template](https://github.com/xonsh/xontrib-template).

## Projects Using Xonsh

Xonsh integrates with many tools:

*   conda and mamba (package managers)
*   Starship (prompt)
*   zoxide (smarter cd)
*   gitsome (Git autocompletion)
*   xxh (SSH)
*   Snakemake (workflow management)
*   any-nix-shell (Nix integration)
*   lix (Nix)
*   x-cmd (Unix tools)
*   rever (software release)
*   Regro autotick bot (Conda-Forge)

**Jupyter Integration:**

*   xontrib-jupyter provides Jupyter-based interactive notebooks
    *   Jupyter and JupyterLab
    *   euporie
    *   Jupytext

## Contribute

Help build the Xonsh community:

*   Contribute code
*   Create xontribs
*   Help with documentation
*   Become a sponsor
*   Spread the word

Join the community on [Zulip](https://xonsh.zulipchat.com/)!