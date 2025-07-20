# Oils: A Modern Upgrade for Your Shell

Oils is a revolutionary project aiming to modernize the Unix shell, offering a superior language and runtime environment. Find the source code at [https://github.com/oils-for-unix/oils](https://github.com/oils-for-unix/oils).

[![Build
Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing shell scripts, providing a robust and compatible shell implementation.
*   **YSH:** A new shell language designed for developers familiar with Python and JavaScript, offering a more modern and user-friendly experience.
*   **Written in Python for ease of modification:** The core logic is written in Python, making the codebase accessible and easy to contribute to.
*   **High-Performance C++ Compilation:** Automatically translates Python code to C++ for performance and efficient execution, without requiring Python as a runtime dependency.
*   **Focus on Compatibility:**  Continuously improving compatibility with existing shell scripts.

## Contributing

Oils welcomes contributions! Here's how you can get involved:

*   **Set up a Dev Build:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build Oils.
*   **Report Issues:** If you encounter any issues during the build process or while using Oils, please let us know by posting on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or by filing an issue on Github.
*   **Start with "Good First Issues":**  Explore and contribute to open issues labeled as "good first issue" on Github ([https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)).
*   **Small contributions are welcome:** Improve spec tests (even without writing code!), influence the design of YSH, or address any issue you find important.

### Quick Start on Linux

1.  Follow the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to get started.
2.  Interact with OSH:

    ```bash
    bash$ bin/osh
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```

3.  Run existing shell scripts:  `bin/osh myscript.sh`
4.  Try out YSH: `bin/ysh`

## Dev Build vs. Release Build

Important: The **developer build** is different from the release tarballs. The [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page describes this difference in detail.

## Important: We Accept Small Contributions!

*   Focus on OSH compatibility by addressing failing [spec tests](https://oils.pub/cross-ref.html#spec-test).
*   Modify code in Python, which is relatively easy to do.
*   Influence the direction of [YSH](https://oils.pub/cross-ref.html#YSH)
*   I aim for a 24 hour response time on Zulip or Github.

## Docs

*   The [Wiki](https://github.com/oils-for-unix/oils/wiki) has detailed developer documentation.
*   End-user documentation is linked from each [release page](https://oils.pub/releases.html).

## Links

*   [Oils Home Page](https://oils.pub/) - Main project website.
*   [Oils Repo Overview](doc/repo-overview.md) - Explains the structure of the project.
*   [README-index.md](README-index.md) - Index to docs for subdirectories.
*   [Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases) - FAQ.