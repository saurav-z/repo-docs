# Oils: A Modern Upgrade for Your Shell

**Oils** offers a powerful upgrade path from bash, providing a better language and runtime for your shell scripting needs. **[Check out the Oils repo on GitHub](https://github.com/oils-for-unix/oils)**.

## Key Features

*   **OSH (Oil Shell):** Runs your existing shell scripts, providing backward compatibility.
*   **YSH (Y-Shell):** A modern shell language designed for Python and JavaScript users, offering a more familiar syntax.
*   **Fast and Efficient:** Written in Python for ease of development, then automatically translated to C++ for performance and a small footprint.
*   **Easy to Contribute:** The codebase is designed to be accessible, with a low barrier to entry for contributors.
*   **Comprehensive Documentation:** Extensive documentation is available to guide both users and developers.

## Getting Started

### For Users

If you want to *use* Oils, don't clone this repo. Visit the latest release: [https://oils.pub/release/latest/](https://oils.pub/release/latest/).

### For Developers: Contributing

Interested in contributing to Oils? Here's how to get started:

1.  Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to set up a development build (Linux recommended).
2.  If you encounter any issues during the build process, report them on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or file an issue on GitHub.
3.  Check the [GitHub Issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for "good first issue" to find tasks to help out with.

#### Quick Start on Linux (After Following Contributing Instructions)

1.  Build and test by following the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) instructions.
2.  Run the Oil shell interactively: `bin/osh`
3.  Try running a shell script: `bin/osh myscript.sh`
4.  Try YSH with: `bin/ysh`

### Dev Build vs. Release Build

The developer build is very different from the release tarball. The [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page details this difference. Release tarballs are linked from the [home page](https://oils.pub/).

### Contributing is Welcome!

Oils welcomes contributions of all sizes!

*   Focus on making OSH compatible; often, this starts with failing [spec tests](https://oils.pub/cross-ref.html#spec-test).
*   Make changes in Python for easier modification.
*   Influence the design of YSH, with ambitious ideas encouraged.

### Response Time

Expect a response from `andychu` on Zulip or GitHub, usually within 24 hours.

## Documentation

*   The [Wiki](https://github.com/oils-for-unix/oils/wiki) provides developer documentation.
*   End-user docs are linked from each [release page](https://oils.pub/releases.html).

## Links

*   **[Oils Home Page](https://oils.pub/)** - Central hub for all things Oils.
*   [Oils Repo Overview](doc/repo-overview.md)
*   FAQ: [The Oils Repo Is Different From the Tarball](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)