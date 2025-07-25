# Oils: Upgrade Your Shell with a Modern Language and Runtime

[Oils](https://github.com/oils-for-unix/oils) is an open-source project aiming to modernize the Unix shell experience, providing a path from bash to a better language and runtime.

**Key Features:**

*   **OSH (Oil Shell):** Runs your existing bash scripts, ensuring compatibility with your current workflows.
*   **YSH (Yet Another Shell):** A new shell language for Python and JavaScript users, offering a more familiar syntax.
*   **Built for Speed and Simplicity:** Written in Python for ease of development and then automatically translated to C++ for high performance.

## Getting Started

To **use** Oils, visit the latest [release](https://oils.pub/release/latest/).  Do not clone this repository for general use.

For developers, the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page guide you to make the dev build.

### Quick Start on Linux (Dev Build)

1.  Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to set up the development environment.
2.  Run the OSH shell: `bin/osh`
3.  Experiment with:
    ```bash
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
4.  Try running a shell script: `bin/osh myscript.sh`
5.  Try YSH: `bin/ysh`

## Contributing

We welcome contributions of all sizes! Here's how you can help:

*   **Fix failing tests:** We often merge failing [spec tests](https://oils.pub/cross-ref.html#spec-test) related to OSH compatibility.
*   **Write code in Python:** The code is written in Python, making it easier to modify. The translation to C++ is automated.
*   **Influence YSH design:** Share your ideas for YSH features and improvements.

## Important Notes for Contributors

*   The developer build is very different from the release tarball; see the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for more details.
*   For OS X users, use the release tarballs from the [home page](https://oils.pub/).

## Contact

Ping `andychu` on Zulip or Github for questions or pull request reviews (expecting a 24-hour response time).

## Documentation

*   [Wiki](https://github.com/oils-for-unix/oils/wiki): Developer documentation
*   [Release Pages](https://oils.pub/releases.html): End-user documentation

## Useful Links

*   [Oils Home Page](https://oils.pub/)
*   [Oils Repo Overview](doc/repo-overview.md)
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)