# Oils: A New Shell for a Better Unix Experience

**Oils** is a forward-thinking project aiming to modernize the Unix shell experience, providing a more robust and user-friendly environment for both new and experienced users.  [Explore the Oils project on GitHub](https://github.com/oils-for-unix/oils).

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing Bash scripts, providing compatibility and a smooth transition.
*   **YSH (Yet Another Shell):** Designed for users familiar with Python and JavaScript, offering a modern scripting language.
*   **Written in Python, Optimized for Speed:** Developed in Python for ease of modification and automatically translated to efficient C++ for performance.
*   **Focus on Usability and Compatibility:**  Addresses the limitations of traditional shells while maintaining compatibility with existing scripts and workflows.
*   **Active Development and Community:** The project is under active development and welcomes contributions from the community.

## Contributing

Oils welcomes contributions! Here's how you can get involved:

*   **Build and Test:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build the developer version.
*   **Report Issues:**  If you encounter any problems, please let us know by posting on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or opening an issue on GitHub.
*   **Contribute Code:** Find a "good first issue" [on GitHub](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and get involved!

### Quick Start on Linux

After following the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page, you'll have a Python program that you can quickly run and change! Try it interactively:

```bash
bash$ bin/osh

osh$ name=world
osh$ echo "hello $name"
hello world
```
* Try running a shell script you wrote with `bin/osh myscript.sh`.
* Try [YSH](https://oils.pub/cross-ref.html#YSH) with `bin/ysh`.

## Developer Build vs. Release Build

The **developer build** is significantly different from the release tarball.  Refer to the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for detailed information.

## Important: We Accept Small Contributions!

Even if you're new to the project, you can make a difference!  Oils welcomes contributions of all sizes:

*   **Spec Tests:**  Help improve OSH compatibility by contributing failing spec tests.
*   **Python Modifications:**  Focus on modifying the Python code, which is relatively easy to understand and change.
*   **YSH Enhancements:**  Influence the design of YSH by implementing new features and ideas.

## Communication

*   Ping `andychu` on Zulip or GitHub if you're waiting for a pull request review or have questions.
*   Expect a response within 24 hours (although there may be delays due to travel).

## Documentation

*   **Wiki:** Comprehensive developer documentation can be found on the [Wiki](https://github.com/oils-for-unix/oils/wiki).
*   **End-User Docs:**  Documentation for end-users is linked from each [release page](https://oils.pub/releases.html).

## Links

*   **Home Page:** [Oils Home Page](https://oils.pub/) - All important links.
*   **Repository Structure:** See the [Oils Repo Overview](doc/repo-overview.md)
*   **Other Resources:**  See [README-index.md](README-index.md), [mycpp/README.md](mycpp/README.md), and [The Oils Repo Is Different From the Tarball](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)