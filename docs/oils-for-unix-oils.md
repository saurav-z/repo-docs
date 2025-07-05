# Oils: Upgrade Your Shell with a Modern Language and Runtime

Oils is an open-source project aiming to modernize the Unix shell by providing an upgrade path from bash, enhancing scripting capabilities and improving performance.  [Learn more at the original repo!](https://github.com/oils-for-unix/oils)

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing shell scripts, offering improved compatibility and features.
*   **YSH (Yet Another Shell):** A new shell language designed for users familiar with Python and JavaScript, providing a more modern scripting experience.
*   **Built for Speed:**  Written in Python for ease of development, then automatically translated to C++ for high performance and small executable size.
*   **Focus on Compatibility:** Strives for robust compatibility with existing bash scripts and commands.
*   **Open Source and Community Driven:**  Actively welcomes contributions and community involvement.

## Getting Started

**For Users:**

To use Oils, download the latest release from:  <https://oils.pub/release/latest/>

**For Developers and Contributors:**

1.  **Clone the Repository:**  `git clone https://github.com/oils-for-unix/oils`
2.  **Build the Development Version:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page.

### Quick Start on Linux (after setting up dev environment)

    bash$ bin/osh

    osh$ name=world
    osh$ echo "hello $name"
    hello world

-   Try running a shell script you wrote with `bin/osh myscript.sh`.
-   Try [YSH](https://oils.pub/cross-ref.html#YSH) with `bin/ysh`.

## Contributing

We welcome contributions! Here's how you can help:

*   **Try the Dev Build:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build Oils.  Report any issues.
*   **Contribute Code:**  Look for [good first issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) on Github, or work on implementing [spec tests](https://oils.pub/cross-ref.html#spec-test)
*   **Influence Design:**  Contribute to the design of [YSH](https://oils.pub/cross-ref.html#YSH) and other features.

### Contribution Tips

*   Contributions are very welcome, even small ones!
*   Code only needs to work in Python initially.  The translation to C++ is handled automatically.
*   For OSH compatibility, failing [spec tests](https://oils.pub/cross-ref.html#spec-test) are often merged.
*   Ping `andychu` on Zulip or Github for pull request reviews or questions.

## Important Considerations

*   **Dev Build vs. Release Build:** Developer builds are different from the release tarballs. The [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page provides details.
*   **OS X Note:** Developer builds may not work on OS X; use the release tarballs.
*   **We value small contributions!**

## Documentation

*   **Wiki:** The [Wiki](https://github.com/oils-for-unix/oils/wiki) has detailed developer documentation.
*   **End-User Docs:** Linked from each [release page](https://oils.pub/releases.html).
*   **Repo Overview:** [Oils Repo Overview](doc/repo-overview.md)
*   **README-index.md:**  Links to docs for some subdirectories.
*   **FAQ:** [Oils 2023 FAQ](https://www.oilshell.org/blog/2023/03/faq.html) / [Why Create a New Unix Shell?](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)

## Links

*   **Home Page:** [Oils Home Page](https://oils.pub/) (All important links)
*   **OSH:** [OSH](https://oils.pub/cross-ref.html#OSH)
*   **YSH:** [YSH](https://oils.pub/cross-ref.html#YSH)
*   **Oils Repo Overview:** [Oils Repo Overview](doc/repo-overview.md)
*   **FAQ:** [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)
*   **Contributing:** [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing)
*   **Blog:** [Oils Blog](https://oils.pub/blog/)
*   **Issue:** [Issues on Github](https://github.com/oils-for-unix/oils/issues)
*   **Zulip Chat:** [oilshell.zulipchat.com](https://oilshell.zulipchat.com/)