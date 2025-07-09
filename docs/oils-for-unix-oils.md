# Oils: Upgrade Your Shell with a Modern Language and Runtime

**Oils** is a project that aims to provide a significantly improved upgrade path from Bash to a better language and runtime, combining the power of shell scripting with modern features. ([View the source code on GitHub](https://github.com/oils-for-unix/oils))

## Key Features

*   **OSH**: Runs your existing shell scripts, ensuring backward compatibility with Bash.
*   **YSH**: Designed for Python and JavaScript users who prefer to avoid shell scripting.
*   **Fast Performance**: Written in Python and automatically translated to C++ for speed and a small footprint.
*   **Easy to Contribute**: The codebase is designed to be easy to modify and contribute to.
*   **Developer-Friendly**: The developer build is easy to set up on Linux.
*   **Active Community**: Get help and contribute on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/).

## Contributing

Contribute to Oils by:

*   Setting up the **dev build** of Oils. See the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for instructions.
*   Reporting any issues you find, either on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or by filing an issue on Github.
*   Tackling a [good first issue](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
*   Contributing failing spec tests.

## Quick Start on Linux

After setting up the dev build (instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page):

1.  Run OSH interactively: `bash$ bin/osh`
2.  Try an example: `osh$ name=world; echo "hello $name"`
3.  Execute your scripts: `bin/osh myscript.sh`
4.  Try YSH: `bin/ysh`

## Important Notes

*   **Developer Build vs. Release Build**: The dev build differs significantly from the release tarballs.  Refer to the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for details. Use release tarballs on OS X.
*   **Small Contributions Welcome**: Oils values contributions of all sizes. Even fixing failing tests helps!
*   **Responsive Maintainer**:  Expect a response within 24 hours. Reach out to `andychu` on Zulip or Github for review requests or questions.

## Docs

Find developer docs on the [Wiki](https://github.com/oils-for-unix/oils/wiki).
End-user docs are linked from each [release page](https://oils.pub/releases.html).

## Links

*   [Oils Home Page](https://oils.pub/) - All the important links.
*   [Oils Repo Overview](doc/repo-overview.md) - Repository Structure.
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases) - FAQ