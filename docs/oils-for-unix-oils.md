# Oils: A Modern Upgrade for Your Shell

**Oils** is a new shell that aims to be a modern upgrade path from bash, offering improved language features and a more robust runtime.

[Visit the original Oils repository](https://github.com/oils-for-unix/oils).

## Key Features

*   **OSH (Oil Shell):** Run your existing shell scripts with improved compatibility and features.
*   **YSH (Oil Shell):** A new shell language for Python and JavaScript users, designed to be a more user-friendly alternative to Bash.
*   **Performance:** Written in Python and translated to C++ for speed and a small footprint. The deployed executable doesn't depend on Python.
*   **Easy Contribution:** The code is written in Python, making it easier to understand and contribute to.
*   **Rapid Development:**  Contributions are often accepted, including failing spec tests, making it a great project for prototyping and learning.

## Getting Started

### For Users

If you want to *use* Oils, don't clone this repo. Instead, visit <https://oils.pub/release/latest/>.

### For Developers

1.  Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build the developer version.
2.  Run interactively:

    ```bash
    bash$ bin/osh
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```

3.  Run your shell scripts with `bin/osh myscript.sh`.
4.  Try [YSH](https://oils.pub/cross-ref.html#YSH) with `bin/ysh`.

## Contributing

Oils welcomes contributions! Here's how you can help:

*   Check out the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for build instructions.
*   Join the `#oil-dev` channel on [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) to ask questions.
*   Browse the [Github issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for good first issues.

## Build vs. Release Builds

The **developer build** is different from the release tarballs, as detailed on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page. Release tarballs are linked from the [home page](https://oils.pub/).

## Small Contributions Welcome

Oils thrives on small contributions.  Your contributions can include:

*   Fixing failing [spec tests](https://oils.pub/cross-ref.html#spec-test).
*   Implementing new features in Python.
*   Influencing the design of [YSH](https://oils.pub/cross-ref.html#YSH).

## Response Time

The maintainer, `andychu`, aims to respond to queries and pull requests within 24 hours. Ping `andychu` on Zulip or Github if you need assistance.

## Documentation

*   The [Wiki](https://github.com/oils-for-unix/oils/wiki) has developer documentation.
*   End-user documentation is linked from each [release page](https://oils.pub/releases.html).

## Links

*   [Oils Home Page](https://oils.pub/)
*   [Oils Repo Overview](doc/repo-overview.md)
*   [README-index.md](README-index.md)
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)
*   [Oilshell Zulip Chat](https://oilshell.zulipchat.com/)
*   [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing)
*   [Oils Releases](https://oils.pub/releases.html)