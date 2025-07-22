# Oils: A New Shell for a Better Unix Experience

Oils is a modern shell aiming to be a seamless upgrade path from bash, offering improved features and a more robust runtime.  [Explore the Oils project on GitHub](https://github.com/oils-for-unix/oils).

## Key Features

*   **OSH (Oil Shell):** Runs your existing bash scripts with improved compatibility and features.
*   **YSH (YAML Shell):** A new shell language for users familiar with Python and JavaScript, designed to be more user-friendly and powerful.
*   **Fast and Efficient:** Written in Python for easy development but translated to C++ for performance and a small footprint.
*   **Easy to Contribute:** The project welcomes contributions, even small ones, with a focus on compatibility testing and language design.

## Getting Started

### Using Oils

If you want to use Oils, don't clone this repository. Instead, visit the [latest release page](https://oils.pub/release/latest/) for stable builds.

### Contributing

Want to contribute to the Oils project?  Here's how:

1.  **Build the Development Version:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build the development version. This should only take a few minutes on a Linux machine.
2.  **Report Issues:** If you encounter any problems, let us know! You can post on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or file an issue on GitHub.
3.  **Start with Good First Issues:** Check out the [good first issues on GitHub](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to get started.  It helps to discuss your plans beforehand.

### Quick Start on Linux (Dev Build)

After following the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page:

1.  Run the interactive shell:
    ```bash
    bash$ bin/osh
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
2.  Try running a shell script: `bin/osh myscript.sh`
3.  Experiment with YSH: `bin/ysh`

### Dev Build vs. Release Build

The development build is different from the release tarballs.  See [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) for details. Release tarballs are linked from the [home page](https://oils.pub/).

## Contributing - Accepted Contributions

Oils embraces various contributions, including those that:

*   Fix failing [spec tests](https://oils.pub/cross-ref.html#spec-test) related to compatibility.
*   Incorporate code changes that work in Python.  The semi-automated translation to C++ is handled separately.
*   Influence the design of [YSH](https://oils.pub/cross-ref.html#YSH).

### Response Time

Expect a response within 24 hours. Feel free to ping `andychu` on Zulip or GitHub.

## Documentation

*   [Wiki](https://github.com/oils-for-unix/oils/wiki) for developer documentation (feel free to edit!).
*   **End-user documentation** is linked from each [release page](https://oils.pub/releases.html).

## Links

*   [Oils Home Page](https://oils.pub/) - All important links
*   [Oils Repo Overview](doc/repo-overview.md) - Repository structure
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases) - FAQ