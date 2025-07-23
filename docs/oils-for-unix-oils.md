# Oils: A New Unix Shell and Runtime for Modern Development

Oils is a project focused on upgrading the Unix shell experience with a more robust language and faster runtime. ([Original Repository](https://github.com/oils-for-unix/oils))

**Key Features:**

*   **OSH:** Runs your existing Bash scripts, offering improved compatibility and features.
*   **YSH:** Designed for Python and JavaScript developers seeking a more modern shell experience.
*   **Fast Performance:** Built with a Python frontend that translates to efficient C++ for speed and minimal dependencies.
*   **Easy to Contribute:** The project welcomes contributions of all sizes, and is great for prototyping!

## Get Started

If you want to use Oils, don't clone this repository directly. Instead, visit the [latest release page](https://oils.pub/release/latest/).

### Quick Start on Linux (Development Build)

After following the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page, you'll have a Python program that you can quickly run and change!

    bash$ bin/osh

    osh$ name=world
    osh$ echo "hello $name"
    hello world

-   Run a shell script with `bin/osh myscript.sh`.
-   Try [YSH](https://oils.pub/cross-ref.html#YSH) with `bin/ysh`.

## Contributing

Oils is actively seeking contributions!

*   **Dev Build:** Build and test Oils using the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page.
*   **Issue Resolution:** Contribute to the project by addressing open issues, including "good first issue" tasks.
*   **Code & Design Influence:** Your contributions can impact the development of OSH and YSH.

## Important Notes:

*   **Development vs. Release Builds:**  Understand the difference between the developer build and the release tarballs (linked from the [home page](https://oils.pub/)).
*   **Small Contributions Welcome:** Even small contributions, such as fixing failing spec tests or influencing YSH design, are highly valuable.
*   **Rapid Response:** You can expect a response on Zulip or GitHub within 24 hours, or sooner.

## Documentation

*   [Wiki](https://github.com/oils-for-unix/oils/wiki): Comprehensive developer documentation.
*   [End-User Docs]: Linked from each [release page](https://oils.pub/releases.html).

## Useful Links

*   [Oils Home Page](https://oils.pub/): Central hub for all things Oils.
*   [Oils Repo Overview](doc/repo-overview.md): Repository structure.
*   [Oils Repo Is Different From the Tarball](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)