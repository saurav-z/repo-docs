# Oils: A Next-Generation Shell for Unix Systems

**Oils is a new shell designed to upgrade your bash scripts and offer a modern scripting experience.** Find the source code on [GitHub](https://github.com/oils-for-unix/oils).

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing bash scripts, providing improved compatibility and performance.
*   **YSH (Yet-Another Shell):** A new language for Python and JavaScript users who prefer a modern syntax and avoid the complexities of traditional shell scripting.
*   **Fast and Efficient:** Written in Python and automatically translated to C++ for speed and a small footprint.  The final executable does not depend on Python.
*   **Easy to Contribute:** The core codebase is written in Python, making it accessible and easy to modify.  
*   **Community Focused:** The maintainers aim for a 24-hour response time for reviews and questions.

## Getting Started

**For Users:**  If you want to use Oils, please visit the latest release page: <https://oils.pub/release/latest/>.  Do not clone this repository for usage.

**For Developers:**  To contribute, follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build the development version.  You can also run the development version of OSH and YSH interactively:

```bash
bash$ bin/osh
osh$ name=world
osh$ echo "hello $name"
hello world
```

Try running a shell script:
```bash
bash$ bin/osh myscript.sh
```

Try YSH:
```bash
bash$ bin/ysh
```

## Contributing

Oils welcomes contributions of all sizes!  Here's how you can get involved:

*   **Build the Dev Version:** Follow the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page instructions to build Oils on your Linux machine.
*   **Report Issues:** If you encounter problems, post on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or file an issue on Github.
*   **Good First Issues:**  Tackle an issue labeled "good first issue" on [Github](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

## Development Build vs. Release Build

The development build (from this repository) is **very different** from the release tarballs.  Refer to the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for details.

## We Accept Small Contributions!

*   **Fix Failing Tests:**  Contribute by fixing failing [spec tests](https://oils.pub/cross-ref.html#spec-test).
*   **Python Focus:** Code primarily in Python.
*   **Influence YSH:**  Shape the future of YSH by contributing new features and ideas.

## Documentation and Resources

*   [Oils Home Page](https://oils.pub/): The central hub for all important links.
*   [Wiki](https://github.com/oils-for-unix/oils/wiki): Developer documentation.
*   [Release Pages](https://oils.pub/releases.html): Documentation for end users.
*   [Oils Repo Overview](doc/repo-overview.md)
*   [README-index.md](README-index.md) for subdirectories.
*   FAQ: [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)
*   [Why Create a New Unix Shell?](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)
*   [Oils 2023 FAQ](https://www.oilshell.org/blog/2023/03/faq.html)