# Oils: A Next-Generation Shell for a Modern Unix Experience

Oils is an ambitious project to upgrade the Unix shell, offering improved scripting and a better runtime experience. Learn more at the [official Oils website](https://oils.pub/).

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing shell scripts with improvements and a cleaner syntax.
*   **YSH (Yet Another Shell):** A new shell designed for Python and JavaScript users, providing a more modern and familiar scripting experience.
*   **Built for Speed and Portability:** Written in Python for easy development and translated to C++ for performance and small size, without external dependencies.
*   **Active Development:** The project is actively developed with a strong community and welcomes contributions.

## Why Oils?

Oils addresses the limitations of traditional shells by offering:

*   **Improved Syntax and Features:**  Modern shell syntax that's easier to read and write.
*   **Enhanced Performance:** Optimized for speed and efficiency, making your scripts run faster.
*   **Better Scripting Experience:** Designed to improve the overall scripting experience for both new and experienced users.

## Getting Started

**If you want to use Oils, do not clone this repo!** Visit <https://oils.pub/release/latest/> to download the latest release.

**For Developers & Contributors:**

1.  **Build the Dev Build:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to set up a development environment.  This typically takes 1-5 minutes on a Linux machine.
2.  **Quick Start (Linux):** After building, run `bin/osh` and try commands like `name=world`, `echo "hello $name"`.  Also, test your scripts with `bin/osh myscript.sh` and try `bin/ysh`.

## Contributing to Oils

We welcome contributions of all sizes!  Whether you're a seasoned programmer or just getting started, there are many ways to help:

*   **Fix Failing Tests:**  Help improve compatibility by fixing failing spec tests.
*   **Implement New Features in YSH:** Shape the future of YSH by contributing new features and improvements.
*   **Improve Documentation:**  Help improve the existing documentation.
*   **Report Issues:**  Let us know about bugs or suggest improvements.

Refer to the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for detailed instructions.

### How to Contribute

*   Find an [issue from Github](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to work on.
*   Discuss your ideas on the `#oil-dev` channel on [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or file an issue on GitHub.
*   Feel free to ask andy about your progress. They aim for a 24-hour response time.

## Documentation

*   **End-User Documentation:** Found on each [release page](https://oils.pub/releases.html).
*   **Developer Documentation:** Available in the [Wiki](https://github.com/oils-for-unix/oils/wiki).

## Links

*   **[Oils Home Page](https://oils.pub/)** - The central hub for all things Oils.
*   **[OSH Documentation](https://oils.pub/cross-ref.html#OSH)**
*   **[YSH Documentation](https://oils.pub/cross-ref.html#YSH)**
*   **[Why Create a New Unix Shell?](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)**
*   **[Oils Repo Overview](doc/repo-overview.md)** - Structure and overview of the repo.
*   **[Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)** - Difference between release and dev builds.

**[View the source code on GitHub](https://github.com/oils-for-unix/oils)**