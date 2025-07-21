# Oils: The Next-Generation Shell for Unix

**Oils is an innovative project aiming to create a modern shell language that offers an upgrade path from bash, while also providing a powerful scripting environment.** ([Original Repo](https://github.com/oils-for-unix/oils))

Oils is designed with the goal of enhancing the scripting experience for both existing shell users and those familiar with languages like Python and JavaScript.  It's written in Python, yet translated to C++ for speed and efficiency, offering a robust and modern shell experience.

## Key Features:

*   **OSH:**  A shell that runs your existing bash scripts.
*   **YSH:**  A new language built with inspiration from Python and JavaScript, designed for modern shell scripting and used in OSH.
*   **Written in Python:** Facilitates easy modification and contribution.
*   **Translated to C++:** Ensures high performance and a small executable size.
*   **Focus on Compatibility:**  Designed to run your existing scripts while offering new features and improvements.

## Contributing

Oils welcomes contributions!  Here's how you can get started:

*   **Build the Dev Version:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build a development version of Oils.
*   **Report Issues:** If you encounter any problems during the build process, please let the team know via [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or by filing an issue on Github.
*   **First Contributions:** Consider grabbing a "good first issue" from the [Github Issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

### Quick Start on Linux

After building the dev version:

1.  Run the interactive shell: `bin/osh`
2.  Test shell scripts: `bin/osh myscript.sh`
3.  Try YSH: `bin/ysh`

**Note:**  The development build is different from the release tarballs.

### Important: Small Contributions Welcome!

Even small contributions are valuable!

*   Contribute failing spec tests.
*   Focus on Python, and the semi-automated translation to C++ will often work.
*   Influence the design of YSH.

## Docs

*   The [Wiki](https://github.com/oils-for-unix/oils/wiki) has developer docs.
*   [End User Docs](https://oils.pub/releases.html) are linked on each release page.

## Links

*   **[Oils Home Page](https://oils.pub/)**: All important links.
*   [Oils Repo Overview](doc/repo-overview.md)
*   [FAQ - The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)