# Oils: A Modern Upgrade Path for Unix Shells

**Oils is a project aiming to modernize the Unix shell experience, providing a powerful and efficient upgrade path from Bash.** [View the source code on GitHub](https://github.com/oils-for-unix/oils).

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

Oils offers two core components:

*   **OSH:**  A compatibility layer that runs your existing shell scripts, making migration easier.
*   **YSH:**  A new shell language designed for users familiar with Python and JavaScript, providing a more modern and user-friendly experience.

**Key Features:**

*   **Upgraded Shell Experience:** Modernizes the Unix shell with improved syntax, features, and performance.
*   **Bash Compatibility (OSH):** Allows you to run existing Bash scripts with minimal changes.
*   **New Language (YSH):** Provides a new shell language inspired by Python and JavaScript for easier adoption.
*   **Fast Performance:** While written in Python for ease of development, Oils is translated to C++ for speed and efficiency.
*   **Open Source & Community Driven:**  Actively developed and maintained with contributions welcome.

## Contributing to Oils

Oils welcomes contributions of all sizes! Here's how you can get involved:

*   **Build & Test:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build the developer version of Oils.
*   **Join the Community:**  Get in touch by posting on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/).
*   **Find an Issue:**  Browse the [Github issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to find a task to contribute.
*   **Small Contributions Welcome:** The project values small contributions such as failing spec tests.

### Quick Start on Linux (Developer Build)

After building the developer version of Oils, test the following to get started:

    bash$ bin/osh

    osh$ name=world
    osh$ echo "hello $name"
    hello world

*   Try running a shell script: `bin/osh myscript.sh`.
*   Test YSH: `bin/ysh`.

### Important: Developer Build vs. Release Build

The developer build is very different from the release tarball.  See the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for details.

### Response Time

If you are waiting for a pull request review, feel free to ping `andychu` on Zulip or Github.  Typically, responses are provided within 24 hours.

## Documentation

*   **Wiki:** The [Wiki](https://github.com/oils-for-unix/oils/wiki) provides developer documentation.
*   **End-User Docs:** End-user documentation is available on each [release page](https://oils.pub/releases.html).

## Links

*   [Oils Home Page](https://oils.pub/) (All important links)
*   [Oils Repo Overview](doc/repo-overview.md)
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)