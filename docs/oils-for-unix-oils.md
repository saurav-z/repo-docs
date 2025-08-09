# Oils: The Next-Generation Unix Shell - Source Code

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

**Oils is an innovative project aiming to upgrade the Unix shell experience with modern features and a more robust foundation.** It offers a smooth transition from Bash, enhancing scripting capabilities while maintaining compatibility.

[View the original repo on GitHub](https://github.com/oils-for-unix/oils)

## Key Features

*   **OSH (Oil Shell):** Runs your existing shell scripts, ensuring compatibility with Bash.
*   **YSH (Yet Another Shell):** Designed for Python and JavaScript users who prefer a more modern shell experience.
*   **Written in Python:** Facilitates rapid development and easier modifications.
*   **Optimized Performance:** Automatically translated to C++ for speed and efficiency, resulting in a fast and small executable.

## Contributing

Oils welcomes contributions! Here's how you can get involved:

*   **Build the Dev Version:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to set up a development build (typically takes 1-5 minutes on a Linux machine).
*   **Reach Out:** If you encounter any issues during setup, report them on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or submit an issue on Github.
*   **Find a Task:** Explore the [Github issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) labeled as "good first issue."
*   **Small Contributions are Encouraged:**  Even minor contributions, such as improving existing tests or addressing failing tests, are valuable.
*   **Python First, C++ Later:** You only need to ensure your code works in Python. The translation to C++ is a separate, often automated, process.
*   **Shape the Future:** Contribute to the design and development of YSH.

### Quick Start on Linux (after following Contributing instructions)

1.  Run `bin/osh` to interact with the shell.
2.  Try a simple command, like `name=world; echo "hello $name"`.
3.  Run your shell scripts using `bin/osh myscript.sh`.
4.  Experiment with YSH by running `bin/ysh`.

## Development vs. Release Builds

**Important:** The developer build you create from this repository is distinct from the release tarballs. The [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page clarifies these differences. For users on OS X, use the release tarballs linked from the [Oils Home Page](https://oils.pub/).

## Docs

*   **Developer Documentation:** The [Wiki](https://github.com/oils-for-unix/oils/wiki) provides extensive developer documentation.
*   **End-User Documentation:**  Found on each [release page](https://oils.pub/releases.html).
*   **Ask for Help:**  If you have questions, ask on Zulip, and someone will help point you to the resources you need.

## Links

*   [Oils Home Page](https://oils.pub/) - Contains all the primary project links.
*   [OSH](https://oils.pub/cross-ref.html#OSH)
*   [YSH](https://oils.pub/cross-ref.html#YSH)
*   [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing)
*   [Oilshell Zulip Chat](https://oilshell.zulipchat.com/)
*   [Why Create a New Unix Shell?](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)