# Oils: A Modern Upgrade to Bash and Shell Scripting

**Oils is a new language and runtime designed to improve shell scripting and provide a more modern experience.** Check out the [Oils GitHub repository](https://github.com/oils-for-unix/oils).

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing bash scripts, offering compatibility with existing shell commands.
*   **YSH (YAML Shell):** A new shell language designed for users familiar with Python and JavaScript, providing a more modern syntax and improved features.
*   **Fast Performance:** Oils is written in Python for ease of development but is automatically translated to C++ for high-performance execution.
*   **Easy to Contribute:** The project is designed to be friendly to contributors, with a low barrier to entry and clear areas for improvement.

## Getting Started

### For Users

If you want to use Oils, the best place to start is the latest release: <https://oils.pub/release/latest/>.  **Do not clone this repository** if you only want to *use* Oils.

### For Developers and Contributors

*   **Build the dev build:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to create a development build of Oils.
*   **Quick Start on Linux (after dev build):**
    *   Run `bin/osh` to start the Oil Shell.
    *   Run `bin/ysh` to start the YSH shell.
    *   Try running a shell script you wrote with `bin/osh myscript.sh`.

## Contributing

Oils welcomes contributions! We are looking for help with all areas of the project.

*   **Contributing is Easy:** Even small contributions, like fixing test failures, are valuable.
*   **Python First:** You primarily work with Python code during development, with the translation to C++ handled automatically.
*   **Influence Design:** You can help shape the future of YSH and the overall project.

### How to Contribute

*   Check out the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for detailed instructions.
*   Consider grabbing an [issue from Github](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
*   Join the `#oil-dev` channel on [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) for help.

## Additional Resources

*   **Home Page:** [https://oils.pub/](https://oils.pub/) (Important Links)
*   **Wiki:** [https://github.com/oils-for-unix/oils/wiki](https://github.com/oils-for-unix/oils/wiki) (Developer Docs)
*   **Releases:** [https://oils.pub/releases.html](https://oils.pub/releases.html) (End-User Docs)
*   **FAQ:** [https://www.oilshell.org/blog/2023/03/faq.html](https://www.oilshell.org/blog/2023/03/faq.html)
*   **Why Create a New Unix Shell?:** [https://www.oilshell.org/blog/2021/01/why-a-new-shell.html](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)