# Oils: A Modern Upgrade Path for Your Shell

**Oils** is a project designed to modernize and improve the Unix shell experience, offering a better language and runtime for your shell scripts. ([See the original repo](https://github.com/oils-for-unix/oils))

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oils Shell):** Runs your existing bash scripts, providing compatibility and a smooth transition.
*   **YSH (Yet Another Shell):** A new shell language designed for users familiar with Python and JavaScript, offering a modern and powerful scripting experience.
*   **Written in Python, Optimized with C++:** The core of Oils is written in Python for ease of development, and then translated to C++ for performance and a small footprint, without relying on Python at runtime.

## Getting Started

### For Users

If you want to **use** Oils, don't clone this repo. Visit the latest release page: <https://oils.pub/release/latest/>

### For Developers and Contributors

To contribute to Oils, follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build the development version.

### Quick Start on Linux (After following Contributing instructions)

1.  Run the interactive shell:
    ```bash
    bin/osh
    ```
2.  Try a simple command:
    ```osh
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
3.  Run a shell script: `bin/osh myscript.sh`
4.  Try YSH: `bin/ysh`

## Contributing to Oils

Oils welcomes contributions of all sizes! The project has a low barrier to entry for developers, making it easy to get involved.

*   **Fix failing spec tests:** Help improve OSH compatibility by contributing test cases.
*   **Influence YSH design:** Share your ideas and suggestions for YSH.
*   **Develop in Python:** Work on the Python code, which is then translated to C++.

**We encourage you to contribute!**

## Important Notes

*   **Dev Build vs. Release Build:** The developer build differs significantly from the release tarballs, particularly regarding OS X compatibility.
*   **Small Contributions Welcome:** Even small contributions are highly valued.

## Docs

*   The [Wiki](https://github.com/oils-for-unix/oils/wiki) provides developer documentation.
*   End-user documentation is linked from each [release page](https://oils.pub/releases.html).

## Links

*   [Oils Home Page](https://oils.pub/)
*   [OSH](https://oils.pub/cross-ref.html#OSH)
*   [YSH](https://oils.pub/cross-ref.html#YSH)
*   [Oils Repo Overview](doc/repo-overview.md)
*   [Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)
*   [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing)
*   [Oilshell Zulip Chat](https://oilshell.zulipchat.com/)