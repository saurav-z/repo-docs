# Oils: A Modern Upgrade for Your Shell

**Oils** is a project to upgrade your shell experience, offering a modern language and runtime that is compatible with existing shell scripts. Explore a better shell with OSH and YSH. [Visit the Oils GitHub Repository](https://github.com/oils-for-unix/oils) to learn more and contribute.

## Key Features

*   **OSH (Oil Shell):** Runs your existing bash scripts with enhanced features and performance.
*   **YSH (Yet Another Shell):** A new shell designed for Python and JavaScript users, offering a more familiar and modern syntax.
*   **Written in Python:** The core of Oils is written in Python, making the code short, maintainable, and easy to contribute to.
*   **Optimized for Performance:**  Automatically translates the Python code to C++ for a fast and small runtime.
*   **Open Source:**  Contribute to Oils and shape the future of the shell!

## Getting Started

If you're interested in *using* Oils, visit the latest releases at <https://oils.pub/release/latest/>. For developers who want to contribute to the project, follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page.

### Quick Start on Linux (Development)

1.  **Follow Contributing Instructions:** Set up your development environment by following the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page.
2.  **Run OSH Interactively:** `bin/osh`
3.  **Test with `echo`:**

    ```bash
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
4.  **Test your scripts:** Try running your existing shell scripts with `bin/osh myscript.sh`.
5.  **Explore YSH:** Experiment with YSH using `bin/ysh`.

## Contributing to Oils

Oils welcomes contributions of all sizes! Whether it's fixing failing spec tests, improving the documentation, or proposing new features for YSH, your contributions are valuable.

*   **Focus on Python:** Develop in Python for ease of modification. The translation to C++ is handled automatically.
*   **Spec Tests:** Contributing spec tests, even if they fail, is a great way to help.
*   **Influence YSH Design:**  Share your ideas and help shape the future of YSH.

## Important Notes

*   **Development vs. Release Builds:** Be aware of the difference between the developer build (from this repository) and the release tarballs. The release tarballs are linked from the [home page](https://oils.pub/). Developer builds may not work on all platforms.
*   **Small Contributions Welcome:** We are looking for people to work on all aspects of the project, no contribution is too small!

## Documentation and Resources

*   **Homepage:** [Oils Home Page](https://oils.pub/) - The central hub for all Oils information.
*   **Wiki:** The [Oils Wiki](https://github.com/oils-for-unix/oils/wiki) contains a wealth of developer documentation.
*   **Releases:** [Release Pages](https://oils.pub/releases.html) - Find documentation for end-users on each release page.

## Community and Support

*   **Zulip Chat:**  Join the Oils community and ask questions on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/).
*   **GitHub Issues:** File issues or browse existing issues on Github.