# Oils: A Modern Upgrade for Your Unix Shell

**Oils is a new Unix shell designed to be a better language and runtime than bash, with compatibility in mind.** ([See the original repo](https://github.com/oils-for-unix/oils))

## Key Features

*   **OSH: Bash Compatibility:** Runs your existing shell scripts, aiming for compatibility with bash.
*   **YSH: A New Language:**  Provides a new shell language that is Python and JavaScript friendly and avoids the pitfalls of shell.
*   **Fast Performance:** Written in Python but translated to C++ for speed and a small footprint.
*   **Easy to Contribute:** The codebase is designed to be accessible, making it easy to contribute, even with small changes.
*   **Rapid Development:** Built with a focus on quickly iterating and improving the shell.

## What is Oils?

Oils offers a practical upgrade path for shell users by providing:

*   **Improved Syntax & Semantics:**  YSH addresses the common issues found in traditional shell scripting.
*   **Stronger Runtime:**  Oils aims to be a more robust and reliable shell environment.

## Contributing to Oils

We welcome contributions!  Get started by:

1.  Following the [Contributing Guide](https://github.com/oils-for-unix/oils/wiki/Contributing) to set up a development build.
2.  If you encounter any issues during setup, report them on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or file an issue on GitHub.
3.  Look for [good first issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) on GitHub.

### Quick Start on Linux (after setting up a development build)

1.  **Start an interactive session:**
    ```bash
    bin/osh
    ```
2.  **Try a simple command:**
    ```osh
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
3.  **Run a shell script:**
    ```bash
    bin/osh myscript.sh
    ```
4.  **Explore YSH:**
    ```bash
    bin/ysh
    ```

### Developer Build vs. Release Build

It's important to note that the developer build differs from the release tarballs. Please refer to the [Contributing Guide](https://github.com/oils-for-unix/oils/wiki/Contributing) for detailed information.  Release tarballs can be found on the [home page](https://oils.pub/).

### Making Small Contributions

We value contributions of all sizes! Even small contributions like fixing spec tests or influencing YSH's design are welcome.

## Documentation

*   **For developers:** Check out the [Wiki](https://github.com/oils-for-unix/oils/wiki) for detailed documentation.  If you have any questions, ask on Zulip!
*   **For end users:**  See the documentation linked from each [release page](https://oils.pub/releases.html).

## Links

*   **Home Page:** [Oils Home Page](https://oils.pub/)
*   **Contributing:** [Contributing Guide](https://github.com/oils-for-unix/oils/wiki/Contributing)
*   **OSH Documentation:** [OSH](https://oils.pub/cross-ref.html#OSH)
*   **YSH Documentation:** [YSH](https://oils.pub/cross-ref.html#YSH)
*   **Why Create a New Unix Shell?:** [Why](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)
*   **FAQ:** [FAQ](https://www.oilshell.org/blog/2023/03/faq.html)
*   **Oils Repo Overview:** [Repo Overview](doc/repo-overview.md)
*   **The Oils Repo Is Different From the Tarball Releases:** [Repo Tarball FAQ](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)