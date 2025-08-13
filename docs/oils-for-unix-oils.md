# Oils: Upgrade Your Shell with a Better Language and Runtime

Oils is a new Unix shell designed as an upgrade path from bash, offering improved features and a more modern programming experience. [View the source code on GitHub](https://github.com/oils-for-unix/oils).

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing bash scripts, ensuring compatibility.
*   **YSH (Your Shell):** A new shell language for Python and JavaScript users, designed for improved usability and modern features.
*   **Fast and Efficient:** Written in Python for ease of development, then automatically translated to C++ for performance.
*   **Open Source:** Actively developed and maintained on GitHub.
*   **Focus on Compatibility:** Designed to be a drop-in replacement for bash, with a focus on running existing scripts.

## Quick Start (Developer Build)

Follow these steps to get started with the developer build:

1.  **Contribute Instructions:** Follow the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) instructions to build the development version of Oils.
2.  **Interactive Shell:** Once built, start an interactive session with `bin/osh`.
3.  **Test Your Scripts:** Run your existing shell scripts with `bin/osh myscript.sh`.
4.  **Experiment with YSH:** Try YSH using `bin/ysh`.

## Contributing

Oils welcomes contributions! Here's how you can help:

*   **Build the Developer Version:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page.
*   **Report Issues:** If you encounter problems, post on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or file an issue on GitHub.
*   **Contribute Code:** Start with a [good first issue](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or contribute a test.
*   **Influence Design:** If you have ideas for YSH, feel free to contribute.

## Important Notes

*   **Developer Build vs. Release Build:** The developer build (built from this repository) is different from the release tarballs. For using Oils, visit the [latest release page](https://oils.pub/release/latest/).
*   **Small Contributions Welcome:** The Oils project is open to many contributions!
    *   Fix failing spec tests.
    *   Focus on Python code changes, with automated C++ translation.
    *   Influence the design of YSH.
*   **Response Time:** Expect a response within 24 hours.  Ping `andychu` on Zulip or GitHub if you need a review!

## Documentation

*   **Developer Documentation:** The [Wiki](https://github.com/oils-for-unix/oils/wiki) contains extensive developer documentation.
*   **End-User Documentation:** Available on each [release page](https://oils.pub/releases.html).

## Links

*   [Oils Home Page](https://oils.pub/) (Main website with all important links)
*   [OSH](https://oils.pub/cross-ref.html#OSH)
*   [YSH](https://oils.pub/cross-ref.html#YSH)
*   [Oils 2023 FAQ](https://www.oilshell.org/blog/2023/03/faq.html)
*   [Why Create a New Unix Shell?](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)
*   [Repo Overview](doc/repo-overview.md)
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)