# Oils: A Modern Upgrade for Your Shell

Oils ([original repo](https://github.com/oils-for-unix/oils)) is a powerful new shell language and runtime designed to improve upon and modernize the traditional Unix shell experience.

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH:** Runs your existing bash scripts, providing a compatible upgrade path.
*   **YSH:** A new language for Python and JavaScript users, designed to be more user-friendly than shell.
*   **Written in Python:** Enables rapid prototyping and easier modification of the code.
*   **Translated to C++:** Provides high performance and small executable size without Python runtime dependencies.

## Getting Started

**For users**:  Visit the [Oils Home Page](https://oils.pub/release/latest/) for the latest releases.

**For contributors**:  Refer to the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page.

### Quick Start on Linux (for Developers)

1.  Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to set up your development environment.
2.  Interact with OSH:

    ```bash
    bin/osh
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
3.  Run a shell script: `bin/osh myscript.sh`
4.  Try YSH: `bin/ysh`

## Contributing

Oils welcomes contributions of all sizes!  See the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for detailed instructions. Key areas for contributions include:

*   **Testing:** Contributing failing spec tests to improve OSH compatibility.
*   **YSH Design:** Influencing the design and implementation of YSH features.

### Response Time

The project maintainer aims for a 24-hour response time for pull request reviews and questions. Don't hesitate to reach out on Zulip or Github if you have any queries!

## Documentation

*   **Developer Docs:** The [Wiki](https://github.com/oils-for-unix/oils/wiki) provides extensive developer documentation.
*   **End-User Docs:**  Linked from each [release page](https://oils.pub/releases.html).

## Useful Links

*   **Home Page:** [Oils Home Page](https://oils.pub/)
*   **Repository Structure:** [Oils Repo Overview](doc/repo-overview.md)
*   **FAQ:**  [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)