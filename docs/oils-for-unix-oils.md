# Oils: Your Upgrade Path From Bash with OSH and YSH

Oils is a modern shell and programming language designed to be a direct upgrade from Bash and a more user-friendly alternative for Python and JavaScript developers. [Check out the Oils project on GitHub!](https://github.com/oils-for-unix/oils)

## Key Features

*   **OSH (Oil Shell):** Runs your existing shell scripts, offering improved compatibility and performance over traditional Bash.
*   **YSH (Yet Another Shell):** A new language tailored for Python and JavaScript users, offering a more familiar syntax and advanced features.
*   **Fast and Efficient:** Written in Python for ease of development, but automatically translated to C++ for speed and minimal dependencies.
*   **Easy to Contribute:** The codebase is designed for easy contribution, especially for Python developers. The focus is on making the process simple.
*   **Active Community:** The project has an active community with quick response times on questions and contributions.

## Why Oils?

Oils aims to provide a smooth transition from Bash while incorporating modern language features and a focus on improved usability. It provides a path for both existing shell script users and those who prefer Python and JavaScript.

## Getting Started

**For Users:**

*   Don't clone this repository. Instead, download the latest release from [the Oils home page](https://oils.pub/release/latest/).

**For Developers (Contributing):**

1.  Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to set up a development build.
2.  Join the `#oil-dev` channel on [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) for questions and support, or open an issue on Github.
3.  Start by tackling [good first issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

### Quick Start on Linux (for Developers)

After setting up the development environment, run:

```bash
bash$ bin/osh
osh$ name=world
osh$ echo "hello $name"
hello world
```

*   Run your scripts with `bin/osh myscript.sh`.
*   Experiment with YSH: `bin/ysh`.

### Dev Build vs. Release Build

Remember the developer build is different from the release tarballs.

## Contributing and Community

Oils welcomes contributions of all sizes! Focus on making your code work in Python first; the translation to C++ is often automated. We are especially looking for failing spec tests.

## Documentation

*   **Home Page:** [Oils Home Page](https://oils.pub/)
*   **Wiki:** [Oils Wiki](https://github.com/oils-for-unix/oils/wiki)
*   **Releases:** [Release Page](https://oils.pub/releases.html)

## Contact

For questions or PR reviews, contact `andychu` on Zulip or Github. Expect a response within 24 hours!