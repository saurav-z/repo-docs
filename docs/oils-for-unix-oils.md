# Oils: A Modern Unix Shell for a Better Shell Experience

**Oils is an innovative project designed to modernize the Unix shell, offering a more powerful and user-friendly experience for both seasoned shell users and those new to the command line.** 

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

Explore the source code for Oils on [GitHub](https://github.com/oils-for-unix/oils).

## Key Features

*   **OSH (Oil Shell):** Runs your existing bash scripts, providing a smoother transition to a more advanced shell.
*   **YSH (Yet-Another Shell):** Designed for Python and JavaScript users, offering a modern shell experience inspired by familiar languages.
*   **Performance:** Written in Python and automatically translated to C++ for speed and efficiency, without requiring Python as a runtime dependency.
*   **Easy to Contribute:**  The code is easy to change because it is written in Python. The translation to C++ is automated, making it easy to test and prototype new features.

## Contributing

Interested in contributing? Here's how:

*   **Build the Dev Build:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build Oils (takes ~1-5 minutes on Linux).
*   **Report Issues:** If you encounter any problems, reach out on the `#oil-dev` channel on [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or create a GitHub issue.
*   **Good First Issues:** Explore the [GitHub issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) marked as "good first issue."

### Quick Start on Linux (after contributing steps above)

1.  Run `bin/osh` in your terminal to test OSH interactively.
2.  Run a shell script: `bin/osh myscript.sh`.
3.  Test YSH: `bin/ysh`.

## Developer Build vs. Release Build

It's important to understand that the **developer build** differs from the release tarballs (linked from the [home page](https://oils.pub/)).  Developer builds are not compatible with OS X. The [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page details these differences.

## We Welcome Small Contributions

Even small contributions are valuable! We encourage contributions in various forms:

*   **Spec Tests:** Help improve OSH compatibility by contributing failing spec tests.
*   **Python Code:** Modify plain Python programs with ease to introduce new features.
*   **YSH Design:** Influence the direction of YSH by experimenting with new ideas, such as nonlinear pipelines.

## Response Time

Expect a response within 24 hours if you have questions or are waiting for a pull request review.

## Documentation

*   **Wiki:**  The [Wiki](https://github.com/oils-for-unix/oils/wiki) has extensive developer documentation. Feel free to edit and ask for assistance on Zulip.
*   **End-User Docs:** Access end-user documentation from each [release page](https://oils.pub/releases.html).

## Links

*   **Home Page:** [Oils Home Page](https://oils.pub/)
*   **Related:**
    *   Repository Structure: [Oils Repo Overview](doc/repo-overview.md)
    *   Subdirectory Docs: See `README-index.md` and files like `mycpp/README.md` for detailed information.
    *   FAQ: [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)