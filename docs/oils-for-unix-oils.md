# Oils: A Modern Upgrade for Your Unix Shell

[Oils](https://github.com/oils-for-unix/oils) is an ambitious project aiming to modernize and improve the Unix shell experience, offering a better language and runtime for your shell scripts.

## Key Features

*   **OSH (Oil Shell):** Runs your existing bash scripts, offering improved performance and features.
*   **YSH (Oil Shell for YAML):** Designed for Python and JavaScript users who want to avoid shell scripting, providing a more familiar syntax and powerful capabilities.
*   **Written in Python, Translated to C++:** The codebase is primarily in Python for ease of development and modification, while being automatically translated to C++ for performance and a small footprint.
*   **Active Community:**  Benefit from a welcoming community eager to help you contribute and improve Oils.

## Getting Started

To *use* Oils, download the latest release from [the releases page](https://oils.pub/release/latest/).

### Quick Start on Linux (Development)

If you're interested in *developing* Oils, follow these steps after following the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page:

1.  Run the development build:
    ```bash
    bin/osh
    ```
2.  Test it out:
    ```osh
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
3.  Run a script: `bin/osh myscript.sh`
4.  Experiment with YSH: `bin/ysh`

## Contributing

Oils welcomes contributions!  The project has a low barrier to entry.

*   **Join the Community:** Engage with the community on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) and the [Github issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
*   **Easy Contributions:** Focus on writing and testing your code in Python, and the semi-automated C++ translation will often just work.  
*   **Influence Design:** Share your ideas and contribute to the design of YSH and other aspects of Oils.

## Important Information

*   **Dev Build vs. Release Build:**  The development build is very different from the release tarball. See the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page for details.
*   **Response Time:**  Expect a response from the maintainer within 24 hours (ping `andychu` on Zulip or Github if you're waiting).

## Documentation and Resources

*   **Home Page:** [Oils Home Page](https://oils.pub/) - All the important links.
*   **Wiki:** [Oils Wiki](https://github.com/oils-for-unix/oils/wiki) - Developer documentation.
*   **Releases:** [Oils Releases](https://oils.pub/releases.html) - End-user documentation linked from each release page.
*   **FAQ:** [Oils 2023 FAQ](https://www.oilshell.org/blog/2023/03/faq.html)