# Oils: A Modern Upgrade for Your Shell (and a New Scripting Language!)

Oils is a project that aims to modernize the Unix shell experience, offering a superior language and runtime that builds upon and improves the foundations of Bash. [Learn more and contribute at the original Oils repository](https://github.com/oils-for-unix/oils).

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml) 
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

**Key Features:**

*   **OSH:** Run your existing shell scripts with improved performance and features.
*   **YSH:** A new scripting language designed for Python and JavaScript users, providing a more familiar and modern syntax.
*   **Performance:** Built with a unique architecture, Oils translates Python code to C++ for speed and efficiency without Python dependencies in the final executable.
*   **Easy to Contribute:** The project is written in Python, making it easier for developers to understand, modify, and contribute.

## Getting Started

If you want to **use** Oils, don't clone this repo. Instead, visit [Oils Releases](https://oils.pub/release/latest/).

### Quick Start on Linux (Dev Build)

To try the developer build, follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page. Then:

```bash
bash$ bin/osh
osh$ name=world
osh$ echo "hello $name"
hello world
```

-   Try running a shell script you wrote with `bin/osh myscript.sh`.
-   Try [YSH](https://oils.pub/cross-ref.html#YSH) with `bin/ysh`.

## Contributing

Oils welcomes contributions! Here's how you can get involved:

*   **Build & Test:** Follow the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to create a developer build, and let us know if you encounter any issues.
*   **First Issues:** Explore the [good first issue](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) on Github for beginner-friendly tasks.
*   **Small Contributions:** Contributions are welcome, even for tasks like fixing failing spec tests.

###  Developer Build vs. Release Build

The **developer build** is **very different** from the release tarball. The [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page describes this difference in detail.

The release tarballs are linked from the [home page](https://oils.pub/).  (Developer builds don't work on OS X, so use the release tarballs on OS X.)

## Important Information

*   **I aim for 24 hour response time.** Please feel free to ping `andychu` on Zulip or Github if you're **waiting** for a pull request review! (or to ask questions)

*   **Docs:** The [Wiki](https://github.com/oils-for-unix/oils/wiki) has many developer docs.

## Links

*   [Oils Home Page](https://oils.pub/) (all important links)
*   [OSH](https://oils.pub/cross-ref.html#OSH)
*   [YSH](https://oils.pub/cross-ref.html#YSH)
*   [Oils Repo Overview](doc/repo-overview.md)
*   [Oils Repo Is Different From the Tarball FAQ](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)