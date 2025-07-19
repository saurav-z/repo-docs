# Oils: A Modern Upgrade for Your Shell & Scripting 

Oils is an ongoing project to build a next-generation shell that provides a modern, safer, and more powerful environment for both shell scripting and interactive use. [Check out the original repo](https://github.com/oils-for-unix/oils).

## Key Features of Oils

*   **OSH (Oil Shell):**  Runs your existing bash scripts, providing a compatibility layer for a smooth transition.
*   **YSH (Oil's Scripting Language):** A new scripting language for users familiar with Python and JavaScript who want to avoid the complexities of traditional shell scripting.
*   **Fast and Efficient:**  Written in Python for ease of development, then automatically translated to C++ for performance and a small footprint.
*   **Focus on Compatibility:**  Continuously updated to ensure compatibility with the existing shell scripting ecosystem.
*   **Open Source and Community-Driven:**  Actively seeking contributions to improve and expand the project.

## Getting Started

For **users**: Download the latest release from [https://oils.pub/release/latest/](https://oils.pub/release/latest/). Do not clone this repository.

For **developers**:

1.  **Contribute:** Explore the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for instructions on building and contributing.
2.  **Join the Community:** Engage with the Oils community on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or by opening issues on GitHub.
3.  **Quick Start on Linux:**
    *   Follow the instructions on the [Contributing][] page.
    *   Run `bin/osh` in your terminal.
    *   Test with `bin/osh myscript.sh` for shell scripts.
    *   Try [YSH][] with `bin/ysh`.

## Contributing Guidelines

Oils welcomes contributions, big or small!  Here's how you can help:

*   **Start Small:**  Contribute by fixing spec tests or making small improvements.
*   **Focus on Python:** Code primarily in Python, the semi-automated translation to C++ happens later.
*   **Influence the Design:**  Share your ideas and influence the development of [YSH][].
*   **Get Help:** Reach out to `andychu` on Zulip or Github for prompt responses and support.

## Resources

*   **Home Page:** [https://oils.pub/](https://oils.pub/) - The central hub for all things Oils.
*   **OSH Documentation:** [https://oils.pub/cross-ref.html#OSH](https://oils.pub/cross-ref.html#OSH)
*   **YSH Documentation:** [https://oils.pub/cross-ref.html#YSH](https://oils.pub/cross-ref.html#YSH)
*   **Contributing Guide:** [https://github.com/oils-for-unix/oils/wiki/Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing)
*   **Frequently Asked Questions:** [https://www.oilshell.org/blog/2023/03/faq.html](https://www.oilshell.org/blog/2023/03/faq.html)
*   **Why a New Unix Shell?** [https://www.oilshell.org/blog/2021/01/why-a-new-shell.html](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)
*   **Oils Repo Overview:** [doc/repo-overview.md](doc/repo-overview.md)
*   **Oils Repo FAQ:** [https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)