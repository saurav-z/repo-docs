# Oils: A Modern Upgrade for Your Unix Shell 

**Oils is a new shell and programming language designed to improve upon the features of Bash and other existing shells.**  ([See the original repo](https://github.com/oils-for-unix/oils))

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH:** Runs your existing Bash scripts, offering a compatibility layer for a smoother transition.
*   **YSH:** A modern language for Python and JavaScript users, designed for scripting and automation, providing an alternative to shell scripting.
*   **Fast and Efficient:** Written in Python and automatically translated to C++, delivering performance without sacrificing ease of development.
*   **Easy to Contribute:** The codebase is designed to be accessible, with many opportunities for contributions of all sizes.

## Why Oils?

Oils provides a modern upgrade path for shell scripting.  It aims to be a better language and runtime.

*   [Why Create a New Unix Shell?](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)
*   [Oils 2023 FAQ](https://www.oilshell.org/blog/2023/03/faq.html)

## Getting Started

If you want to **use** Oils, don't clone this repo. Visit <https://oils.pub/release/latest/> for the latest releases.
[The Oils Repo Is Different From theTarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases).

### Quick Start on Linux (Developer Build)

1.  Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build Oils.
2.  Run the interactive OSH shell: `bin/osh`
3.  Try a simple command:

    ```bash
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
4.  Run a shell script: `bin/osh myscript.sh`.
5.  Try YSH: `bin/ysh`.

Let us know if any of these things don't work! [The continuous
build](https://op.oilshell.org/) tests them at every commit.

### Dev Build vs. Release Build

The **developer build** is **very different** from the release tarball.  The [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page describes this difference in detail.

The release tarballs are linked from the [home page](https://oils.pub/).  (Developer builds don't work on OS X, so use the release tarballs on OS X.)

## Contributing

Oils welcomes contributions! The bar to contribute is very low, making it a great project for newcomers.

*   **Small Contributions Welcome:** Focus on areas like failing spec tests, which help improve compatibility, or influencing the design of YSH.
*   **Python First:** Code is easily modifiable as it's primarily written in Python.
*   **Get Involved:** Start by grabbing an issue tagged as a "good first issue" on [Github](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).  Let us know what you're thinking before you get too far.
*   **Response Time:**  The primary developer, `andychu`, aims for a 24-hour response time on Zulip or Github, so don't hesitate to reach out with questions or for pull request reviews.

### Docs

*   [Wiki](https://github.com/oils-for-unix/oils/wiki) has many developer docs
*   Docs for **end users** are linked from each [release
    page](https://oils.pub/releases.html).

## Links

*   [Oils Home Page](https://oils.pub/)
*   [OSH](https://oils.pub/cross-ref.html#OSH)
*   [YSH](https://oils.pub/cross-ref.html#YSH)
*   Related:
    *   Repository Structure: See the [Oils Repo Overview](doc/repo-overview.md)
    *   The [README-index.md](README-index.md) links to docs for some
        subdirectories.  For example, [mycpp/README.md](mycpp/README.md) is pretty
        detailed.
    *   FAQ: [The Oils Repo Is Different From the Tarball][repo-tarball-faq]

[repo-tarball-faq]: https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases