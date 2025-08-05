# Oils: A Modern Upgrade for Your Shell

**Oils is a new Unix shell built to improve upon and extend the capabilities of bash, offering a better language and runtime for your shell scripting needs.**

[![Build
Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

[Visit the Oils project on GitHub](https://github.com/oils-for-unix/oils)

## Key Features

*   **OSH (Oil Shell):**  Runs your existing shell scripts, aiming for enhanced compatibility and performance.
*   **YSH (Oil Shell):** A modern shell language for users familiar with Python and JavaScript, providing a more expressive and user-friendly experience.
*   **Written in Python:**  The core language is written in Python for ease of development and modification.
*   **Optimized for Performance:** Automatically translated to C++ for fast and efficient execution without Python dependencies in the final executable.
*   **Focus on Compatibility:** Designed to be a robust upgrade to your existing shell environment.

## Getting Started

*   **For Users:**  To use Oils, visit the latest release page: <https://oils.pub/release/latest/>.  Don't clone this repository to *use* Oils.
*   **For Developers:** See the [Contributing](#contributing) section below.

## Contributing

We welcome contributions to the Oils project!  Here's how you can get involved:

*   **Build the Dev Version:** Follow the instructions on the [Contributing][] page to build the development version of Oils.
*   **Report Issues:** If you encounter problems or have suggestions, please create an issue on Github or post on the `#oil-dev` channel of [oilshell.zulipchat.com][].
*   **First Contributions:**  Check out the [good first issue](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) list.

### Quick Start on Linux (for developers)

1.  Follow the instructions on the [Contributing][] page.
2.  Interact with the shell:
    ```bash
    bin/osh
    ```
    ```osh
    osh$ name=world
    osh$ echo "hello $name"
    hello world
    ```
3.  Try running a shell script: `bin/osh myscript.sh`
4.  Try [YSH][] with `bin/ysh`.

### Dev Build vs. Release Build

The **developer build** differs significantly from the release tarballs. The [Contributing][] page details these differences. Release tarballs are linked from the [home page][home-page]. (Developer builds don't work on OS X, so use the release tarballs on OS X.)

### Small Contributions Welcome!

Oils thrives on community contributions.  Even small contributions are highly valued!

*   We often merge failing [spec tests](https://oils.pub/cross-ref.html#spec-test).  Tests alone help!
*   You only need to make your code work **in Python**. The translation to C++ is separate.
*   You can help design [YSH][].

**Response Time:**  Expect a response within 24 hours. Ping `andychu` on Zulip or Github if you have a pull request or question.

## Documentation

*   The [Wiki](https://github.com/oils-for-unix/oils/wiki) has developer documentation.
*   Docs for **end users** are linked from each [release page](https://oils.pub/releases.html).
*   **Home Page**: [Oils Home Page][home-page]

## Resources

*   [Oils Home Page][home-page]
*   [Oils Repo Overview][repo-overview]
*   FAQ: [The Oils Repo Is Different From the Tarball][repo-tarball-faq]
*   [Oils Shell Blog](https://oils.pub/blog/)
*   [Oils Shell Zulip Chat](https://oilshell.zulipchat.com/)
*   [YSH][]: Link to YSH Documentation/Homepage
*   [OSH][]: Link to OSH Documentation/Homepage

[Contributing]: https://github.com/oils-for-unix/oils/wiki/Contributing
[oilshell.zulipchat.com]: https://oilshell.zulipchat.com/
[home-page]: https://oils.pub/
[OSH]: https://oils.pub/cross-ref.html#OSH
[YSH]: https://oils.pub/cross-ref.html#YSH
[repo-overview]: doc/repo-overview.md
[repo-tarball-faq]: https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases