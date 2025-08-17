# Oils: Upgrade Your Shell with a Modern Language and Runtime

**Oils** is an ambitious project aiming to modernize the Unix shell experience, offering an improved language and runtime for shell scripting. ([View the source code on GitHub](https://github.com/oils-for-unix/oils/))

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):**  Run your existing Bash scripts with improved performance and features.
*   **YSH (Yet Another Shell):** A new shell language for Python and JavaScript users, designed for modern shell scripting.
*   **Python-Based Development:**  The core logic is written in Python, making it easier to modify and contribute to.
*   **High-Performance Runtime:**  Oils translates Python code to C++ for a fast and small deployed executable, without requiring a Python runtime.
*   **Actively Developed & Community Supported:**  Get involved with a project that's always improving!

##  Why Oils?

Oils seeks to modernize the shell, a core part of the Unix ecosystem.  It aims to improve compatibility with existing scripts while providing a more modern and powerful programming environment. Check out the [Oils 2023 FAQ](https://www.oilshell.org/blog/2023/03/faq.html) for more details.

## Getting Started

*   **For Users:**  To *use* Oils, see the latest releases at: <https://oils.pub/release/latest/>

*   **For Contributors:**  See the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for instructions on building and contributing.
    *   Join the community:  Ask questions and get involved on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or open an issue on Github.
    *   Find "good first issues":  Contribute to the project by finding issues labeled "good first issue" in the [issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) section.

### Quick Start on Linux (For Developers)

After following the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page:

```bash
bin/osh  # Try interactively
osh$ name=world
osh$ echo "hello $name"
hello world
```

*   Run your scripts with `bin/osh myscript.sh`.
*   Try [YSH](https://oils.pub/cross-ref.html#YSH) with `bin/ysh`.

##  Contribution Guidelines

The project welcomes contributions of all sizes!

*   **Focus on Python:** Modify the Python source code, which is easy to change. The translation to C++ is semi-automated.
*   **Spec Tests:**  Contribute by merging failing [spec tests](https://oils.pub/cross-ref.html#spec-test) to improve OSH compatibility.
*   **Influence YSH:**  Share ideas and contribute to the design of [YSH](https://oils.pub/cross-ref.html#YSH).

##  Support and Documentation

*   **Response Time:**  Expect a response within 24 hours from the project lead (andychu) on Zulip or Github.
*   **Wiki:** The [Wiki](https://github.com/oils-for-unix/oils/wiki) has detailed developer documentation.
*   **User Documentation:**  End-user documentation is available on each [release page](https://oils.pub/releases.html).

##  Links

*   [Oils Home Page](https://oils.pub/) - Central hub for information.
*   [Oils Repo Overview](doc/repo-overview.md)
*   [README-index.md](README-index.md)
*   [mycpp/README.md](mycpp/README.md)
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)