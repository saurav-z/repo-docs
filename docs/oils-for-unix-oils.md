# Oils: A Modern Upgrade for Your Unix Shell

**Oils** is a project designed to modernize the Unix shell experience, providing a powerful upgrade path from bash to a more advanced language and runtime. [Visit the Oils GitHub repository](https://github.com/oils-for-unix/oils).

## Key Features

*   **OSH (Oil Shell):** Runs your existing shell scripts, ensuring compatibility with your existing workflow.
*   **YSH (YAML Shell):** A new shell language for users of Python and JavaScript, designed to provide a more familiar and efficient scripting experience.
*   **Fast Performance:** Written in Python, but automatically translated to C++ for optimal speed and a small footprint. The deployed executable does not require Python.
*   **Easy Contribution:** The codebase is designed to be accessible, with contributions often focused on compatibility and new feature development.

## Getting Started

To **use** Oils, visit <https://oils.pub/release/latest/>.

### Quick Start on Linux (for Developers)

After following the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page, you'll have a Python program you can run and change.

1.  Run the interactive shell:
    ```bash
    bin/osh
    ```
2.  Try some commands:
    ```osh
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
3.  Run a shell script you wrote with `bin/osh myscript.sh`.
4.  Try [YSH](https://oils.pub/cross-ref.html#YSH) with `bin/ysh`.

## Contributing

We welcome contributions! Here's how you can get involved:

*   **Build the Dev Version:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build the development version (typically takes 1-5 minutes on Linux).
*   **Report Issues:** If you encounter any problems during the build process, let us know on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or by filing an issue on GitHub.
*   **Good First Issues:** Find a good first issue on GitHub: [Open Issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
*   **Small Contributions Welcome:** The bar to contribution is low! You can focus on fixing failing spec tests, which contribute to compatibility, or help shape the future of YSH.

## Resources

*   [Oils Home Page](https://oils.pub/) - Main website with links to releases, documentation, and more.
*   [Oils Blog](https://oils.pub/blog/) - Stay up-to-date with project news and developments.
*   [Wiki](https://github.com/oils-for-unix/oils/wiki) - Contains detailed developer documentation.
*   [Oils Repo Overview](doc/repo-overview.md) - See the repository structure.
*   [FAQ](https://www.oilshell.org/blog/2023/03/faq.html)
*   [Why Create a New Unix Shell?](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)

## Contact

For questions or pull request reviews, ping `andychu` on Zulip or GitHub, who usually responds within 24 hours.