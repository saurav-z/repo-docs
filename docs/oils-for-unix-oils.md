# Oils: A Modern Upgrade for Your Shell

Oils is a new Unix shell offering an upgrade path from Bash, designed to be faster, more reliable, and easier to use. [(View the source code on GitHub)](https://github.com/oils-for-unix/oils)

Key Features:

*   **OSH:** Runs your existing Bash scripts, ensuring compatibility.
*   **YSH:**  A new language for Python and JavaScript users, designed to avoid the pitfalls of shell scripting.
*   **Modern Design:**  Written in Python for ease of development and maintenance, then automatically translated to C++ for speed and efficiency.
*   **Focus on Compatibility:**  Continuously tested to ensure compatibility with existing shell scripts and standards.
*   **Active Community:**  Open to contributions, with a low barrier to entry, and a responsive maintainer.

## Why Choose Oils?

Oils is designed to address the shortcomings of traditional shells like Bash, offering a modern and improved experience for both developers and end-users. Whether you're looking for a better scripting language or a more robust runtime, Oils provides a compelling solution.

## Getting Started

If you want to use Oils, don't clone this repository. Instead, visit the [latest release page](https://oils.pub/release/latest/).

To contribute to the project, start by following the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build the developer version of Oils.

### Quick Start on Linux (Developer Build)

After setting up your development environment:

1.  Run `bin/osh` to test OSH.
2.  Try `bin/osh myscript.sh` to run a shell script.
3.  Run `bin/ysh` to test YSH.

## Contributing to Oils

We welcome contributions of all sizes!

*   **Good First Issues:**  Start by tackling issues tagged as "good first issue" on [GitHub](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
*   **Spec Tests:** Contribute by fixing failing [spec tests](https://oils.pub/cross-ref.html#spec-test).
*   **Python-First Development:** Modify the Python code, and the translation to C++ will often work automatically.

## Important Notes

*   **Developer Build vs. Release Build:** The developer build differs significantly from the release tarballs. See the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for details.
*   **Small Contributions Welcome:** The bar to contribution is low. The project welcomes all types of contributions.

## Support & Communication

*   For questions and discussions, use the `#oil-dev` channel on [oilshell.zulipchat.com](https://oilshell.zulipchat.com/).
*   If you're waiting for a pull request review, ping `andychu` on Zulip or GitHub.

## Documentation

*   Developer documentation is available on the [Wiki](https://github.com/oils-for-unix/oils/wiki).
*   End-user documentation can be found on each [release page](https://oils.pub/releases.html).
*   The [README-index.md](README-index.md) links to documentation for certain subdirectories.

## Additional Links

*   [Oils Home Page](https://oils.pub/)
*   [Oils Repo Overview](doc/repo-overview.md)
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)
*   [Oils 2023 FAQ](https://www.oilshell.org/blog/2023/03/faq.html)
*   [Why Create a New Unix Shell?](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)