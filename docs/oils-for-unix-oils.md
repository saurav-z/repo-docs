# Oils: Upgrade Your Shell with a Modern, Fast, and Flexible Language

[Oils](https://github.com/oils-for-unix/oils) is your upgrade path from bash, providing a better language and runtime for your shell scripting needs.

## Key Features:

*   **OSH (Oil Shell):**  Runs your existing shell scripts, offering compatibility with bash while introducing improvements.
*   **YSH (Oil Scripting Language):** Designed for Python and JavaScript users, providing a modern scripting experience.
*   **Fast Performance:** Written in Python and automatically translated to C++ for speed and efficiency.
*   **Easy to Contribute:** The project's Python codebase is easy to modify, making it ideal for prototyping and contributions.
*   **Active Development:**  The project is actively maintained with a focus on community contributions and improvements.

## Getting Started

If you want to **use** Oils, don't clone this repo.  Instead, visit <https://oils.pub/release/latest/>.

## Contributing

Contribute to the development of Oils by following these steps:

1.  **Build the Dev Version:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build the Oils development version.
2.  **Report Issues:** Report any issues or ask questions on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or by filing an issue on Github.
3.  **Contribute Code:** Find an [issue from Github](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or implement a feature, remembering that you can often contribute by simply writing tests.

## Quick Start on Linux (Dev Build)

After following the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page, try the following:

```bash
bash$ bin/osh

osh$ name=world
osh$ echo "hello $name"
hello world
```

Try running a shell script you wrote with `bin/osh myscript.sh` or try [YSH](https://oils.pub/cross-ref.html#YSH) with `bin/ysh`.

## Important Notes:

*   The **developer build** is **very different** from the release tarball.  See [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) for details.
*   The [Oils Home Page](https://oils.pub/) contains all the important links.