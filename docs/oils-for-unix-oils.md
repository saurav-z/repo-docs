# Oils: Upgrade Your Shell with a Modern Language and Runtime

[Oils](https://github.com/oils-for-unix/oils) provides an upgrade path from bash to a better language and runtime, built for modern shell scripting.

## Key Features

*   **OSH (Oil Shell):**  Runs your existing bash scripts, offering compatibility with your current workflow.
*   **YSH (YAML Shell):** A new shell language designed for users familiar with Python and JavaScript, providing a more modern and intuitive syntax.
*   **Fast and Efficient:** Written in Python and automatically translated to C++ for performance and a small footprint.
*   **Easy to Contribute:** The project welcomes contributions, even small ones, with clear guidelines and responsive maintainers.
*   **Developer-Friendly:** Provides a straightforward development experience with a Python-based codebase and clear documentation.

## Getting Started

### Development Build

To get started with Oils development:

1.  Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to set up the development environment.
2.  Run `bin/osh` and `bin/ysh` after building to test the shell.

### Quickstart Example

```bash
bash$ bin/osh

osh$ name=world
osh$ echo "hello $name"
hello world
```

## Contributing

Oils is actively developed and welcomes contributions! Even small contributions are valued.

*   Fix failing spec tests.
*   Improve the YSH language design.
*   Contribute to the documentation.
*   Ask questions and get involved on [oilshell.zulipchat.com](https://oilshell.zulipchat.com/).
*   Grab an [issue from Github](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

## Important Notes

*   **Developer Build vs. Release Build:** The development build is different from the release tarball. Refer to the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for more details.
*   **Where to get the latest release:**  Visit the [Oils Home Page](https://oils.pub/release/latest/) to use the latest releases.

## Documentation and Resources

*   **Home Page:** [Oils Home Page](https://oils.pub/) - Central hub for all important links.
*   **Wiki:**  [Oils Wiki](https://github.com/oils-for-unix/oils/wiki) - Developer documentation.
*   **Releases:** [Oils Releases](https://oils.pub/releases.html) - User documentation.
*   **FAQ:** [Oils FAQ](https://www.oilshell.org/blog/2023/03/faq.html) - Answers to common questions.
*   **Why Create a New Unix Shell?** [Why Create a New Unix Shell?](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)

**Thank you for your interest in Oils!**