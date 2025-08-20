# Oils: Upgrade Your Shell Experience with OSH and YSH

**Oils is a new Unix shell designed to be a modern upgrade path from Bash, offering improved scripting capabilities and a more user-friendly experience.**  [View the source code on GitHub](https://github.com/oils-for-unix/oils).

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing Bash scripts, ensuring compatibility with your current workflow.
*   **YSH (Your Shell):** A new shell designed for Python and JavaScript users, providing a more modern and accessible scripting language.
*   **Fast Performance:** Written in Python and automatically translated to C++ for optimized speed and a small footprint.
*   **Easy to Contribute:** The codebase is designed to be easy to modify, with a focus on Python development and semi-automated C++ translation.

## Quick Start on Linux (Development Build)

After setting up your development environment (see [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing)), try these commands:

```bash
bin/osh
```

```osh
name=world
echo "hello $name"
hello world
```

*   Run your shell scripts with `bin/osh myscript.sh`.
*   Try YSH with `bin/ysh`.

## Contributing

Oils welcomes contributions of all sizes! Here's how you can get involved:

*   **Build the Dev Version:**  Follow the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) guide.
*   **Find Issues:** Explore the [good first issue](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) issues on GitHub.
*   **Small Contributions Welcome:** Focus on Python code and consider contributing failing spec tests to OSH compatibility.

## Important Notes

*   **Developer Build vs. Release Build:**  The development build is different from the release tarballs (linked from the [Oils Home Page](https://oils.pub/)).  Consult the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for details.
*   **Feedback:**  If you have questions or issues, reach out on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or file an issue on Github.

## Documentation and Resources

*   [Oils Home Page](https://oils.pub/):  All key links are here.
*   [Oils Wiki](https://github.com/oils-for-unix/oils/wiki): Developer documentation.
*   [Oils Repo Overview](doc/repo-overview.md)
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)
*   **End-user documentation** can be found on each [release page](https://oils.pub/releases.html).