# Oils: A Modern Upgrade for Your Shell - From Bash to a Better Language

[Oils](https://github.com/oils-for-unix/oils) is a new shell designed to be a modern upgrade for the Unix shell, providing improved features and a smoother experience for developers. It's built on the principle of incremental improvement, starting with compatibility with existing shell scripts and evolving to a more powerful and user-friendly language.

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml) 
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing shell scripts, ensuring backward compatibility.
*   **YSH (Oil Shell):** A new shell language designed for Python and JavaScript users who want to avoid the complexities of traditional shell scripting.
*   **Built with Python & Translated to C++:**  Developed in Python for ease of modification and then automatically translated to C++ for speed and efficiency, without a dependency on Python at runtime.
*   **Open Source and Actively Developed:**  Community-driven project with a welcoming approach to contributions of all sizes.

## Getting Started

If you want to **use** Oils, please visit the latest release at <https://oils.pub/release/latest/>. Do not clone this repository directly if you only want to use Oils.

### Quick Start (Developer Build - for contributing)

1.  Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to set up the development environment.
2.  Run OSH interactively: `bin/osh`
3.  Try a simple command: `osh$ echo "hello world"`
4.  Run a shell script: `bin/osh myscript.sh`
5.  Try YSH: `bin/ysh`

## Contributing

Oils is designed to be approachable for new contributors. We welcome contributions of all sizes, even if it's just improving documentation.

*   **Easy to Contribute:**  The project welcomes small contributions like fixing failing spec tests or improving existing Python code.
*   **Influence Design:** Opportunities to shape the future of YSH and other features.
*   **Rapid Feedback:**  The maintainer (andychu) aims for a 24-hour response time on pull requests and questions.

### Contributing Resources

*   [Contributing Guide](https://github.com/oils-for-unix/oils/wiki/Contributing)
*   [Good First Issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
*   [Zulip Chat](https://oilshell.zulipchat.com/)

## Documentation

*   [Wiki](https://github.com/oils-for-unix/oils/wiki) for developer documentation.
*   [Release pages](https://oils.pub/releases.html) for end-user documentation.
*   [Oils Home Page](https://oils.pub/) for all key links.