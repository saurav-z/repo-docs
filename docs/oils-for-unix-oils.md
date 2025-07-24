# Oils: Upgrade Your Shell with a Modern Language and Runtime

**Oils** is an open-source project providing an upgrade path from bash to a more powerful and versatile shell language and runtime, built for modern needs. [Visit the Oils GitHub Repository](https://github.com/oils-for-unix/oils) for more information and to contribute.

## Key Features

*   **OSH:** Runs your existing bash shell scripts, ensuring compatibility.
*   **YSH:** A new shell language designed for Python and JavaScript users who want to avoid traditional shell scripting.
*   **Fast Performance:** Written in Python and automatically translated to C++ for speed and efficiency.
*   **Easy to Contribute:** The core codebase is written in Python, making it accessible for developers of all levels.

## Getting Started

### Contributing

Oils welcomes contributions! To get started:

*   Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build the development version.
*   If you encounter any issues, report them on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or file an issue on GitHub.
*   Explore the [good first issue](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) list for beginner-friendly tasks.

### Quick Start on Linux (Dev Build)

After setting up the dev build (see [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing)), try:

1.  Run the interactive shell: `bin/osh`
2.  Test with a simple script: `bin/osh myscript.sh`
3.  Explore YSH: `bin/ysh`

## Developer Build vs. Release Build

The developer build is distinct from the release tarballs. The [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page explains these differences in detail. Release tarballs are linked from the [home page](https://oils.pub/).

## Contributing Guidelines

We encourage contributions of all sizes!  Here's what makes contributing easy:

*   Focus on Python:  Code is written in Python, simplifying modifications.
*   Compatibility:  Merge failing [spec tests](https://oils.pub/cross-ref.html#spec-test) for OSH.
*   Design Influence: Influence the design of YSH.

## Communication and Support

*   **Response Time:** Expect a response from the core team within 24 hours. Ping `andychu` on Zulip or Github for pull request reviews or questions.
*   **Docs:**  Refer to the [Wiki](https://github.com/oils-for-unix/oils/wiki) and ask questions on Zulip for any confusion.
*   **End-User Docs:** Found on each [release page](https://oils.pub/releases.html).

## Resources

*   [Oils Home Page](https://oils.pub/) - Your central hub for all important links.
*   [Oils Repo Overview](doc/repo-overview.md) - Information on the structure of the repository.
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases) - Explains the difference between the repo and tarball releases.