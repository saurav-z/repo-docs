# Oils: Your Upgrade Path from Bash with a Modern Shell

[Oils](https://github.com/oils-for-unix/oils) is a project that aims to provide a superior alternative to Bash, offering a better language and runtime for shell scripting.

[![Build
Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml) <a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing shell scripts, providing compatibility with Bash.
*   **YSH (Yet Another Shell):** A modern shell language designed for Python and JavaScript users, offering a more familiar syntax.
*   **Built for Performance:** Written in Python for ease of development, then translated to C++ for speed and efficiency without a Python dependency in the final executable.
*   **Easy to Contribute:** The project welcomes contributions, even small ones, focusing on OSH compatibility and improvements to YSH.

## Quick Start (Linux)

1.  Follow the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) instructions to build Oils.
2.  Run the interactive OSH shell:  `bin/osh`
3.  Test a shell script: `bin/osh myscript.sh`
4.  Experiment with YSH: `bin/ysh`

## Contributing to Oils

Oils welcomes contributions of all sizes!  Here's how you can help:

*   **Build and Test:** Follow the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) instructions to build and test Oils.
*   **Report Issues:**  Let us know if you encounter any problems by posting in the `#oil-dev` channel on [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or by opening an issue on Github.
*   **Contribute Code:**  Pick up a "good first issue" from the [Github issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). Don't hesitate to discuss your ideas before starting.
*   **Focus on Python:** Contributions are often made directly in Python. The translation to C++ is handled separately.

## Developer Build vs. Release Builds

*   The developer build is for contributing and development, offering a quick and easy way to change code.
*   Release tarballs are linked from the [Oils home page](https://oils.pub/).
*   Developer builds don't work on OS X; use the release tarballs on OS X.

## Documentation and Support

*   **Wiki:** Explore the detailed [Wiki](https://github.com/oils-for-unix/oils/wiki) for developer documentation. Feel free to edit!
*   **End-User Docs:** Find user-focused documentation on each [release page](https://oils.pub/releases.html).
*   **Community:** For any questions or confusion, ask for help on [oilshell.zulipchat.com](https://oilshell.zulipchat.com/).

## Important Links

*   **Home Page:** [https://oils.pub/](https://oils.pub/)
*   **OSH:** [https://oils.pub/cross-ref.html#OSH](https://oils.pub/cross-ref.html#OSH)
*   **YSH:** [https://oils.pub/cross-ref.html#YSH](https://oils.pub/cross-ref.html#YSH)
*   **Contributing:** [https://github.com/oils-for-unix/oils/wiki/Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing)
*   **Oils Repo Overview:** [doc/repo-overview.md](doc/repo-overview.md)
*   **Oils Repo Is Different From the Tarball Releases:** [https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)