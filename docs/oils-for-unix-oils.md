# Oils: Upgrade Your Shell with a Modern Language and Runtime

**Oils** is an open-source project aiming to upgrade the Unix shell experience, offering a powerful and efficient alternative to traditional shell scripting. [Explore the Oils Repository](https://github.com/oils-for-unix/oils).

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing bash scripts, offering enhanced performance and features.
*   **YSH (Oils Shell):**  A new shell designed for Python and JavaScript users, providing a more modern and familiar syntax.
*   **Fast and Efficient:** Written in Python but automatically translated to C++ for speed and small executable size, without Python dependencies in the deployed version.

## Get Started

If you want to **use** Oils, don't clone this repo. Instead, visit <https://oils.pub/release/latest/>.

### Quick Start on Linux (Development Build)

Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to make the dev build.

Once you have the development build, try these commands interactively:

    bash$ bin/osh

    osh$ name=world
    osh$ echo "hello $name"
    hello world

*   Run a shell script with `bin/osh myscript.sh`.
*   Try [YSH](https://oils.pub/cross-ref.html#YSH) with `bin/ysh`.

## Contributing to Oils

Oils welcomes contributions! Here's how you can get involved:

*   **Build & Test:** Try making the dev build as described in the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page.
*   **Report Issues:** If you encounter any problems, let the team know through the `#oil-dev` channel on [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or by filing an issue on GitHub.
*   **Contribute Code:** Grab an issue on [GitHub](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and let the team know what you're thinking before you get too far.

### Contribution Benefits
*   **Small Contributions Welcome:** The bar to contribution is very low. The team is open to accepting contributions of failing tests.
*   **Work in Python:** You only have to make your code work in Python, which is easy to modify. The automated translation to C++ is a separate step.
*   **Influence YSH:** You can influence the design of [YSH](https://oils.pub/cross-ref.html#YSH). If you have an itch to scratch, be ambitious!

## Documentation

*   **Developer Documentation:** The [Wiki](https://github.com/oils-for-unix/oils/wiki) contains extensive developer documentation.
*   **End-User Documentation:** Found on each [release page](https://oils.pub/releases.html).

## Links

*   [Oils Home Page](https://oils.pub/) -  Your central resource for all things Oils.
*   [OSH](https://oils.pub/cross-ref.html#OSH)
*   [YSH](https://oils.pub/cross-ref.html#YSH)
*   [Oils Repo Overview](doc/repo-overview.md) - A guide to the repository structure.
*   [FAQ](https://www.oilshell.org/blog/2023/03/faq.html)