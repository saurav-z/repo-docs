# Oils: A Modern Upgrade for Your Shell & Scripts

Oils is an open-source project aiming to modernize the Unix shell experience, providing a better language and runtime for shell scripting. [Visit the original repo](https://github.com/oils-for-unix/oils).

## Key Features

*   **OSH (Oil Shell):** Runs your existing bash scripts, ensuring compatibility.
*   **YSH (Yet Another Shell):** A modern shell language for Python and JavaScript users, offering a more familiar syntax.
*   **Built for Speed:** Written in Python for ease of development, then translated to C++ for high performance and a small footprint.
*   **Easy to Contribute:**  The project welcomes small contributions, particularly those focused on test improvements and compatibility.
*   **Active Development:**  Regularly updated with continuous builds and a responsive maintainer.

## Getting Started (Developer Build)

To begin contributing, follow these steps:

1.  **Setup:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page.
2.  **Interact with OSH:** Start the shell with `bin/osh` and try running shell scripts with `bin/osh myscript.sh`.
3.  **Explore YSH:** Experiment with `bin/ysh` to experience the new language.

## Contributing Guide

*   **Find an Issue:** Browse [Github Issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks.  Good first issues are available.
*   **Communicate:** Share your ideas before diving in.
*   **Focus on Python:** Implement features in Python; the C++ translation is largely automated.
*   **Influence Design:** Your contributions to YSH can directly impact its evolution.

## Important Notes

*   **Developer vs. Release Builds:** The developer build is distinct from the release tarballs.
*   **For Users:** To use Oils, get the latest release from <https://oils.pub/release/latest/>.

## Support & Docs

*   **Community:** Join the `#oil-dev` channel on [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) for questions and discussions.
*   **Docs:** For end-users, find the documentation on the [release page](https://oils.pub/releases.html). Developers find documentation in the [Wiki](https://github.com/oils-for-unix/oils/wiki).

## Additional Resources

*   [Oils Home Page](https://oils.pub/) - Main project website.
*   [Oils Repo Overview](doc/repo-overview.md) - Repository structure.
*   [README-index.md](README-index.md) - Links to documentation for some subdirectories.
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases) - Important information regarding the differences between the repo and tarball releases.