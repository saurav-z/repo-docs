# Oils: Upgrade Your Shell with OSH and YSH

**Oils is a new Unix shell, providing a modern and efficient upgrade path from Bash, with both OSH (for existing shell scripts) and YSH (a Python/JavaScript-friendly alternative).**

[![Build
Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

Explore the future of shell scripting with Oils, designed for both compatibility and innovation.

**Key Features:**

*   **OSH (Oil Shell):**  Runs your existing Bash scripts, offering improved performance and features.
*   **YSH (Yank Shell):** A modern shell language for Python and JavaScript developers, designed to be more intuitive and powerful.
*   **Performance & Efficiency:**  Written in Python and translated to C++ for speed and a small footprint.
*   **Easy to Contribute:** The codebase is designed to be easy to understand and modify, with a low barrier to entry for contributors.

**Getting Started**

*   **For Users:**  If you want to use Oils, download the latest release from <https://oils.pub/release/latest/>.  Do not clone this repository for general usage.
*   **Developer Build:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build Oils.

**Contributing**

Oils welcomes contributions!  Here's how you can get involved:

*   **Build the Dev Version:** The instructions are on the [Contributing][] page.
*   **Report Issues:**  Let us know if you encounter any problems by filing an issue on Github or posting on the `#oil-dev` channel of [oilshell.zulipchat.com][].
*   **Contribute Code:**  Consider starting with a [good first issue](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to get started.
*   **Small contributions welcome:** Focus on testing or specific features that you want.
*   **Responsive Maintainer:** The project maintainer aims for a 24-hour response time.  Ping `andychu` on Zulip or Github if you need help.

**Quick Start (Dev Build on Linux)**

1.  Follow the [Contributing][] page.
2.  Try it interactively: `bin/osh`
3.  Try a shell script you wrote: `bin/osh myscript.sh`
4.  Try YSH: `bin/ysh`

**Important Notes:**

*   The developer build is different from the release tarballs.
*   The release tarballs are available from the [Oils Home Page][home-page].

**Docs**

*   **End-User Documentation:**  Linked from each [release page](https://oils.pub/releases.html).
*   **Developer Documentation:**  See the [Wiki](https://github.com/oils-for-unix/oils/wiki) for developer docs.  Feel free to edit them.

**Links**

*   **[Oils Home Page][home-page]**:  Your central hub for all things Oils.
*   **[Oils Repository](https://github.com/oils-for-unix/oils)**: View the source code and contribute.
*   [OSH]: [https://oils.pub/cross-ref.html#OSH]
*   [YSH]: [https://oils.pub/cross-ref.html#YSH]
*   [Oils 2023 FAQ][faq-2023] / [Why Create a New Unix Shell?][why]

[home-page]: https://oils.pub/
[OSH]: https://oils.pub/cross-ref.html#OSH
[YSH]: https://oils.pub/cross-ref.html#YSH
[faq-2023]: https://www.oilshell.org/blog/2023/03/faq.html
[why]: https://www.oilshell.org/blog/2021/01/why-a-new-shell.html
[Contributing]: https://github.com/oils-for-unix/oils/wiki/Contributing
[oilshell.zulipchat.com]: https://oilshell.zulipchat.com/