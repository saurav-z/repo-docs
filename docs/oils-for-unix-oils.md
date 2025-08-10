# Oils: A Modern Upgrade to Bash for a Better Shell Experience

**Oils** is a project dedicated to building a next-generation shell that's compatible with your existing Bash scripts while offering a more robust and modern language and runtime. Explore the Oils source code on [GitHub](https://github.com/oils-for-unix/oils).

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing Bash scripts with improved performance and features.
*   **YSH (Yet Another Shell):** A new shell language inspired by Python and JavaScript, designed for modern scripting.
*   **Fast and Efficient:** Written in Python and automatically translated to C++ for speed and a small footprint.
*   **Easy to Contribute:** The codebase is designed to facilitate contributions, even for small improvements and fixes.
*   **Active Community:** Engage with the developers and community on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/).

## Getting Started

If you are interested in *using* Oils, download the latest release: <https://oils.pub/release/latest/>.

### Quick Start for Developers (Linux)

Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build Oils.  Then, try these commands:

```bash
bash$ bin/osh

osh$ name=world
osh$ echo "hello $name"
hello world
```

*   Run your shell scripts with `bin/osh your_script.sh`.
*   Experiment with [YSH](https://oils.pub/cross-ref.html#YSH) using `bin/ysh`.

## Contributing to Oils

Oils welcomes contributions of all sizes!

*   **Start with the Dev Build:** Build and test the dev build.
*   **Good First Issues:** Explore [Good First Issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) on GitHub.
*   **Focus on Python:** Contributions often involve modifying Python code, making them easier to understand and contribute to.
*   **Influence YSH:** Shape the design and functionality of YSH through your contributions.

### Code Style
*   Use `black` for formatting (see CONTRIBUTING.md)

## Development Builds vs. Release Builds

The developer build is **distinct** from the release tarballs.  The [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page offers details on the differences. Release builds are found on the [home page](https://oils.pub/).

## Docs

*   **Developer Documentation:** The [Wiki](https://github.com/oils-for-unix/oils/wiki) offers extensive developer documentation.
*   **User Documentation:** Access end-user documentation via each [release page](https://oils.pub/releases.html).
*   **If you're confused:** Ask on Zulip!

## Links

*   [Oils Home Page](https://oils.pub/) - Main website with all important links.
*   [Oils Repo Overview](doc/repo-overview.md)
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)
*   [Oils Shell Blog](https://oils.pub/blog/)