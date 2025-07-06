# Oils: The Unix Shell Upgrade Path

**Oils** is a modern take on the Unix shell, offering a powerful upgrade path from bash and a new language for Python and JavaScript users. ([View on GitHub](https://github.com/oils-for-unix/oils))

## Key Features

*   **OSH (Oil Shell):** Runs your existing shell scripts, improving compatibility and performance.
*   **YSH (Yet Another Shell):**  A new language for users who want to avoid shell scripting, with a focus on Python and JavaScript users.
*   **Fast and Efficient:** Written in Python for ease of development, but automatically translated to C++ for speed and small size; the deployed executable doesn't require Python.
*   **Open Source and Collaborative:**  Actively developed with a low barrier to contribution. The project welcomes contributions from users of all experience levels.

## Getting Started

### Quick Start on Linux (Developer Build)

1.  Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to set up a developer build.
2.  Run the interactive shell: `bin/osh`
3.  Try a simple command: `osh$ echo "hello $name"`

### Using Oils (Release Build)

If you want to use Oils for your projects, download the latest release from [oils.pub/release/latest/](https://oils.pub/release/latest/) instead of cloning the repo.

## Contributing

We welcome contributions!  Here's how you can help:

*   **Build the project:** Follow the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) instructions.
*   **Report issues:**  Post on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or file an issue on Github.
*   **Fix issues:**  Pick up a "good first issue" from [Github](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

**Important:** The developer build is different from the release tarball.  See the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for details.

## Development Details

*   **Written in Python, Translated to C++:** The project uses Python for rapid development and translation to C++ for optimized performance.
*   **Focus on Compatibility:** The project aims for full compatibility with existing bash scripts through OSH.
*   **Influence YSH Design:**  Contribute to the design and implementation of YSH.

## Documentation and Resources

*   **Home Page:**  [oils.pub](https://oils.pub/)
*   **OSH Documentation:** [oils.pub/cross-ref.html#OSH](https://oils.pub/cross-ref.html#OSH)
*   **YSH Documentation:** [oils.pub/cross-ref.html#YSH](https://oils.pub/cross-ref.html#YSH)
*   **Contributing:** [https://github.com/oils-for-unix/oils/wiki/Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing)
*   **FAQ:** [oils.pub/blog/2023/03/faq.html](https://oils.pub/blog/2023/03/faq.html)
*   **Wiki:** [https://github.com/oils-for-unix/oils/wiki](https://github.com/oils-for-unix/oils/wiki)
*   **Developer Docs:**  See the [Oils Repo Overview](doc/repo-overview.md), and subdirectory READMEs (e.g., [mycpp/README.md](mycpp/README.md)).
*   **Release Builds**: [oils.pub/releases.html](https://oils.pub/releases.html)