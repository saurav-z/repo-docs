# Oils: The Modern Shell for Unix (and Beyond!)

[Oils](https://github.com/oils-for-unix/oils) is a next-generation shell, designed to be a superior replacement for Bash, with an improved language and runtime.  It offers a path to a better shell experience.

**Key Features:**

*   **OSH:** Runs your existing Bash scripts, ensuring compatibility.
*   **YSH:** A modern shell language for users familiar with Python and JavaScript, offering a new way to script.
*   **Built for Speed:** Written in Python for ease of development, but translated to C++ for a fast and efficient runtime.
*   **Easy to Contribute:** The project welcomes contributions, even small ones like fixing spec tests.

## Getting Started

If you want to use Oils, visit the latest release: <https://oils.pub/release/latest/>. This repository contains the source code and information for developers. 

**Note:** The development build is distinct from the release tarballs. Refer to the [Oils Wiki](https://github.com/oils-for-unix/oils/wiki) for more information.

## Contributing to Oils

We encourage contributions of all sizes!

*   **Build and Test:** Follow the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) instructions to set up a development environment.
*   **Join the Community:** Ask questions and discuss development on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or open an issue on Github.
*   **Find Good First Issues:** Explore [open issues on Github](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to get started.

### Quick Start (Linux)

1.  Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page.
2.  Run the interactive shell: `bash$ bin/osh`
    ```bash
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
3.  Execute a script: `bin/osh myscript.sh`
4.  Try the YSH interpreter: `bin/ysh`

## Developer Resources

*   **Docs:** The [Wiki](https://github.com/oils-for-unix/oils/wiki) is full of developer documentation.
*   **End-User Docs:** Find documentation for end-users on each [release page](https://oils.pub/releases.html).

## Important Links

*   [Oils Home Page](https://oils.pub/)
*   [Oils Repo Overview](doc/repo-overview.md)
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)