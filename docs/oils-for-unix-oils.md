# Oils: A Modern Upgrade for Your Shell - Faster, Safer, and More Powerful

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

[Oils](https://oils.pub/) is a powerful and modern shell project, offering an upgrade path from bash and a better language for scripting. Developed in Python and translated to C++ for performance, Oils aims to improve scripting while maintaining compatibility.

**Key Features:**

*   **OSH (Oil Shell):** Runs your existing bash scripts, ensuring compatibility.
*   **YSH (Oil Shell):**  A new shell language designed for users familiar with Python and JavaScript, aiming for improved safety and usability.
*   **Fast and Efficient:**  Written in Python but translated to C++ for optimized performance and a small footprint.
*   **Open Source:** [View the Oils source code on GitHub](https://github.com/oils-for-unix/oils).
*   **Easy to Contribute:**  The codebase is designed to be accessible, and contributions of all sizes are welcome.

## Getting Started

For users looking to *use* Oils, please visit the [latest release page](https://oils.pub/release/latest/).  This repository is for development and contribution.

### Quick Start for Developers (Linux)

1.  Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to set up your development environment.
2.  Run the interactive shell: `bash$ bin/osh`
3.  Experiment with OSH:
    ```bash
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
4.  Run your scripts: `bin/osh myscript.sh`.
5.  Try YSH: `bin/ysh`

## Contributing to Oils

We welcome contributions! Here's how you can get involved:

*   **Build the Dev Version:**  Follow the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) instructions to build Oils.
*   **Report Issues:**  Let us know if you encounter any problems by posting on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or by opening an issue on GitHub.
*   **Contribute Code:** Check out [good first issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) on Github.  We especially welcome improvements to test coverage and features in YSH.
*   **Influence the Design:**  Your contributions can help shape the future of YSH and Oils.

### Development Build vs. Release Build

The developer build differs significantly from the release tarballs. The [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page provides detailed information. Release tarballs are available from the [home page](https://oils.pub/).

### Small Contributions are Appreciated

Even small contributions are valuable! We prioritize improvements to compatibility, especially through failing tests. You can also influence the design and direction of YSH.

### Response Time

Expect a response within 24 hours. Feel free to ping `andychu` on Zulip or Github if you're waiting for a review.

## Documentation

*   [Oils Home Page](https://oils.pub/): Central hub for all things Oils.
*   [Wiki](https://github.com/oils-for-unix/oils/wiki): Detailed developer documentation.
*   [End User Docs](https://oils.pub/releases.html): Linked from each release page.
*   [Oils Repo Overview](doc/repo-overview.md)
*   [README-index.md](README-index.md): Links to docs for some subdirectories, such as [mycpp/README.md](mycpp/README.md)
*   FAQ: [The Oils Repo Is Different From the Tarball](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)