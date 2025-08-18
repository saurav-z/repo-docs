# Oils: A Modern Upgrade for Your Unix Shell

**Oils is a new Unix shell that provides a modern upgrade path from bash, offering improved features and a smoother experience for both shell script users and Python/JavaScript developers.** Explore the source code on [GitHub](https://github.com/oils-for-unix/oils).

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH:** Runs your existing shell scripts, providing compatibility with bash.
*   **YSH:** A new shell language designed for Python and JavaScript users who prefer avoiding bash.
*   **Fast Performance:** Written in Python but automatically translated to C++ for speed and efficiency. The deployed executable is independent of Python.
*   **Easy to Contribute:** The project welcomes contributions, especially for improving OSH compatibility and shaping the future of YSH.

## Project Overview

Oils aims to modernize the Unix shell experience. It comprises two key components: OSH, for compatibility with existing shell scripts, and YSH, a new shell language designed for a wider audience. The project's architecture leverages Python for rapid prototyping and automated translation to C++ for performance.

## Contributing

Oils welcomes contributions from the community! Get involved by:

*   **Building the Dev Build:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to create a developer build.
*   **Reporting Issues:** Let us know if you encounter any problems! Post on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or file an issue on Github.
*   **Picking Up "Good First Issues":** Explore [open issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) labeled as "good first issue" to get started.
*   **Contributing Tests:** Improving OSH compatibility often involves adding and improving tests.

### Quick Start on Linux

1.  Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page.
2.  Run the following commands interactively:

    ```bash
    bin/osh
    ```

    ```osh
    name=world
    echo "hello $name"
    hello world
    ```

3.  Run a shell script with `bin/osh myscript.sh`.
4.  Try `bin/ysh`.

## Developer Build vs. Release Build

The **developer build** is **different** from the release tarball.  The [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page describes this difference in detail. Release tarballs are linked from the [home page](https://oils.pub/).

## Important: We Accept Small Contributions!

Oils is perfect for prototyping and experimentation.

*   **OSH Compatibility:**  Contribute by improving test coverage.
*   **Python-Based:**  Contributions can be made in Python. The semi-automated translation to C++ is a separate step.
*   **Influence YSH Design:** Shape the future of YSH.

## Docs

*   The [Wiki](https://github.com/oils-for-unix/oils/wiki) has many developer docs.
*   Docs for **end users** are linked from each [release page](https://oils.pub/releases.html).

## Links

*   [Oils Home Page](https://oils.pub/)
*   [OSH](https://oils.pub/cross-ref.html#OSH)
*   [YSH](https://oils.pub/cross-ref.html#YSH)
*   [Oils Repo Overview](doc/repo-overview.md)
*   [README-index.md](README-index.md)
*   [FAQ](https://www.oilshell.org/blog/2023/03/faq.html)

```

Key improvements and explanations:

*   **SEO Optimization:**  Keywords like "Unix shell," "bash upgrade," "new shell language," "shell scripting," "Python," and "C++" are included naturally throughout the text.  The use of descriptive headings and bullet points helps with readability and SEO.
*   **Hook:**  The one-sentence hook immediately grabs the reader's attention and explains the core purpose of Oils.
*   **Clear Structure:**  Headings break up the content and make it easy to scan.
*   **Concise Language:**  The information is presented in a straightforward and easy-to-understand manner.
*   **Call to Action:** The contributing section actively encourages involvement.
*   **Internal and External Links:** Links are provided to key project resources, including the original repo and supporting documentation.
*   **Emphasis on Benefits:** The "Key Features" section highlights the advantages of using Oils.
*   **Removed redundancies:** Simplified and consolidated information from the original README.
*   **Clarified Quick Start:** Provides more specific, actionable steps for getting started.
*   **Added Context:** Provides better descriptions.
*   **Improved Formatting:** Formatting has been improved for better readability.
*   **Includes Gitpod badge**