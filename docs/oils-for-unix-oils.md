# Oils: A Modern Upgrade for Your Shell

**Oils is a new language and runtime, designed as an upgrade path from bash, offering improved features and performance.**

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

[View the Oils Source Code on GitHub](https://github.com/oils-for-unix/oils)

## Key Features

*   **OSH:** Runs your existing shell scripts, ensuring compatibility.
*   **YSH:** A new language for Python and JavaScript users who want to avoid shell scripting.
*   **Fast and Efficient:** Written in Python and automatically translated to C++ for speed and a small footprint.
*   **Easy to Contribute:** The Python codebase is designed to be easy to understand and modify, making it ideal for prototyping and contribution.

## Why Oils?

Oils aims to solve the limitations of traditional shell scripting. [Learn more about the motivations](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html) behind its creation and [read the FAQs](https://www.oilshell.org/blog/2023/03/faq.html).

## Getting Started (Development)

1.  Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to make the **dev build** of Oils.
2.  If you encounter any issues, please let us know by posting on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or filing an issue on Github.
3.  Start by exploring [good first issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

### Quick Start on Linux (after building)

    bash$ bin/osh

    osh$ name=world
    osh$ echo "hello $name"
    hello world

*   Try running a shell script with `bin/osh myscript.sh`.
*   Try [YSH](https://oils.pub/cross-ref.html#YSH) with `bin/ysh`.

## Development vs. Release Builds

It's important to note the differences between the **developer build** and the release tarballs. The [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page provides a detailed explanation. Release tarballs are linked from the [home page](https://oils.pub/).

## Contributing Guidelines

We encourage small contributions!

*   Focus on improving OSH compatibility or [YSH](https://oils.pub/cross-ref.html#YSH) features.
*   Code changes are primarily implemented in Python.
*   [Nonlinear pipelines](https://github.com/oils-for-unix/oils/issues/843) are a great example of what can be contributed.
*   Expect a response within 24 hours! Ping `andychu` on Zulip or Github for PR reviews or questions.

## Documentation

*   The [Wiki](https://github.com/oils-for-unix/oils/wiki) offers extensive developer documentation.
*   End-user documentation is available on each [release page](https://oils.pub/releases.html).

## Important Links

*   [Oils Home Page](https://oils.pub/) - All essential information.
*   [Oils Repo Overview](doc/repo-overview.md)
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)