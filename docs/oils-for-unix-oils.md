# Oils: A Modern Upgrade for Your Shell

**Oils is a new Unix shell designed to be a modern and practical upgrade from Bash, offering improved scripting capabilities and a smoother user experience.** [Learn more about Oils](https://github.com/oils-for-unix/oils)

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH: Bash Compatibility:** Runs your existing shell scripts, making migration easy.
*   **YSH: A Modern Scripting Language:** Designed for Python and JavaScript users who prefer a more modern syntax than Bash.
*   **Fast and Efficient:**  Written in Python and automatically translated to C++ for performance and a small footprint, without relying on a Python runtime.
*   **Easy to Contribute:** The codebase is designed to be accessible and easy to modify, with a focus on Python for rapid prototyping.

## Getting Started

*   **For Users:** Don't clone this repository to use Oils.  Instead, download the latest release from [the Oils releases page](https://oils.pub/release/latest/).  See also [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)
*   **For Developers:**  To contribute, follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build the development version.

### Quick Start on Linux (for Developers)

After setting up the development environment:

1.  Run `bin/osh` to try OSH interactively.
2.  Run a shell script with `bin/osh myscript.sh`.
3.  Try YSH with `bin/ysh`.

## Contributing

Oils welcomes contributions of all sizes!  Here's how you can get involved:

*   **Get the Dev Build Working:** Follow the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) instructions to create the development build. If you encounter any issues, please report them.
*   **Find an Issue:**  Look for "good first issue" labels on the [Oils issue tracker](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).  Discuss your approach before you start.
*   **Focus on Python:**  The initial development is in Python, making modifications easy.  The translation to C++ is largely automated.
*   **Influence YSH:**  Help shape the future of YSH by contributing features and improvements.
*   **Small Contributions Welcome:**  Even submitting failing spec tests (that highlight areas for improvement) is helpful.

### Response Time

Expect a response within 24 hours.  Ping `andychu` on Zulip or GitHub if you need a review or have questions.

## Documentation

*   **Wiki:** The [Oils Wiki](https://github.com/oils-for-unix/oils/wiki) provides extensive developer documentation.
*   **End-User Docs:**  Documentation for end-users is linked from each [release page](https://oils.pub/releases.html).

## Links

*   **[Oils Home Page](https://oils.pub/)** - The central hub for all Oils information.
*   **[Oils Repo Overview](doc/repo-overview.md)** -  Repository structure
*   **[README-index.md](README-index.md)** - Docs for subdirectories
*   **FAQ:** [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)

---
**[View the Oils Source Code on GitHub](https://github.com/oils-for-unix/oils)**