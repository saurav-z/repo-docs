# Oils: A Modern Upgrade for Your Shell

**Oils** is a project offering a new shell built on Python, designed to be a more user-friendly and powerful alternative to traditional shells like Bash. [Visit the Oils GitHub repository](https://github.com/oils-for-unix/oils) for more information.

**Key Features:**

*   **OSH:** Runs existing shell scripts, ensuring compatibility with your current workflow.
*   **YSH:** A new shell syntax for those coming from Python or JavaScript.
*   **Python-Based:** Code is written in Python for ease of modification and rapid development.
*   **C++ Optimization:** Automatically translates Python code to C++ for speed and efficiency.
*   **Active Development:**  Benefit from a project with many contributors.

## How to Get Started

If you want to use Oils, please visit the [Oils release page](https://oils.pub/release/latest/) instead of cloning this repository.

### Contributing

We welcome contributions!  If you want to contribute, here are the steps:

*   Follow the instructions on the [Contributing page](https://github.com/oils-for-unix/oils/wiki/Contributing) to create a dev build.
*   Ask on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or file an issue on Github if there are any problems.
*   Contribute by grabbing an [issue from Github](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
*   Check the blog for [Many Ideas](https://oils.pub/blog/)

### Quick Start on Linux (Dev Build)

After following the instructions on the [Contributing page](https://github.com/oils-for-unix/oils/wiki/Contributing), you will have a Python program to quickly run and change!

1.  Try it interactively:
    ```bash
    bash$ bin/osh

    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
2.  Run a shell script: `bin/osh myscript.sh`.
3.  Try [YSH](https://oils.pub/cross-ref.html#YSH): `bin/ysh`.

### Important Differences: Dev Build vs. Release Build

Please note that the **developer build** is **very different** from the release tarball. More details can be found on the [Contributing page](https://github.com/oils-for-unix/oils/wiki/Contributing).

### Contributing is Easy!

*   Focus on Python: Only contribute to Python.
*   Influence design: You can influence the design of [YSH](https://oils.pub/cross-ref.html#YSH).

### Response Time

Expect a response within 24 hours! You can reach out to `andychu` on Zulip or Github.

## Documentation

Comprehensive documentation is available:

*   [Wiki](https://github.com/oils-for-unix/oils/wiki): Developer documentation.
*   [Release Pages](https://oils.pub/releases.html): Documentation for end-users.
*   See the [Oils Home Page](https://oils.pub/) has all the important links.
*   Related:
    *   Repository Structure: See the [Oils Repo Overview](doc/repo-overview.md)
    *   The [README-index.md](README-index.md) links to docs for some
        subdirectories.  For example, [mycpp/README.md](mycpp/README.md) is pretty
        detailed.
    *   FAQ: [The Oils Repo Is Different From the Tarball](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)