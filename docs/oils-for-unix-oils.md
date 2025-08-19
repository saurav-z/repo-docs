# Oils: Upgrade Your Shell! 🚀

**Oils is an open-source project aiming to modernize and improve the Unix shell experience, offering a superior language and runtime for both existing and future shell scripts.**

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

Explore the source code for Oils: a project that aims to upgrade the shell with improved language features and performance.  This repository hosts the development code for Oils. **[View the Oils project on GitHub](https://github.com/oils-for-unix/oils)**.

## Key Features

*   **OSH:** Runs your existing bash scripts, ensuring compatibility and a smooth transition.
*   **YSH:** A modern shell language designed for Python and JavaScript users, providing a familiar and powerful scripting environment.
*   **Performance:** Written in Python for ease of development but automatically translated to C++ for fast and small executables.
*   **Active Development:** Continuously improving with regular updates and an active community.

## Getting Started

*   **Using Oils:**  If you want to *use* Oils, visit the latest release page at <https://oils.pub/release/latest/>, *not* the source code repository. See also: [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases).
*   **Quick Start (Dev Build on Linux):** After following the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) instructions:
    ```bash
    bash$ bin/osh
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
    Try running a shell script with `bin/osh myscript.sh` or explore `bin/ysh`.

## Contributing

Oils welcomes contributions of all sizes!  The bar to contribution is low:

*   **First-time contributors:** Consider starting with the [good first issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
*   **OSH Compatibility:** Contribute by merging failing spec tests to improve OSH compatibility.
*   **YSH Development:** Influence the design of YSH. Propose and implement new features.

### Development Build vs. Release Build

*   **Developer Build:** Focuses on quick iteration and modification of the Python code. Follow the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for instructions.
*   **Release Build:** The stable, distributable version, linked from the [home page](https://oils.pub/).

### Important:  We Accept Small Contributions!

Your contributions, even small ones, are highly valued.  Reach out to `andychu` on Zulip or Github if you need help or a pull request review.

## Documentation

*   **Developer Docs:** The [Wiki](https://github.com/oils-for-unix/oils/wiki) provides in-depth documentation for developers.
*   **End-User Docs:**  Available on each [release page](https://oils.pub/releases.html).

## Links

*   [Oils Home Page](https://oils.pub/) - The central hub for information.
*   [OSH Documentation](https://oils.pub/cross-ref.html#OSH)
*   [YSH Documentation](https://oils.pub/cross-ref.html#YSH)
*   [Why Create a New Unix Shell?](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)
*   [Oils 2023 FAQ](https://www.oilshell.org/blog/2023/03/faq.html)
*   [Oils Repo Overview](doc/repo-overview.md)
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)