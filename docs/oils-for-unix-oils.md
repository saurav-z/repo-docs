# Oils: Your Upgrade Path From Bash to a Better Shell

**Oils** is a modern shell language and runtime designed to be a significant upgrade over Bash, offering improved features, performance, and a more approachable syntax, and it's available on [GitHub](https://github.com/oils-for-unix/oils).

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml) <a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing Bash scripts with enhanced features and performance.
*   **YSH (Yet Another Shell):** A new shell language for Python and JavaScript developers, designed for modern use cases.
*   **Fast Performance:** Written in Python and automatically translated to C++ for speed and efficiency.
*   **Easy to Contribute:** The codebase is designed to be easy to modify.  Small contributions are welcome!

## Getting Started

If you want to **use** Oils, don't clone this repo.  Instead, visit
<https://oils.pub/release/latest/>.

For the **developer build** follow the instructions on the [Contributing][] page.

## Quick Start on Linux (Developer Build)

After following the instructions on the [Contributing][] page, you'll have a
Python program that you can quickly run and change!  Try it interactively:

    bash$ bin/osh

    osh$ name=world
    osh$ echo "hello $name"
    hello world

- Try running a shell script you wrote with `bin/osh myscript.sh`.
- Try [YSH][] with `bin/ysh`.

Let us know if any of these things don't work!  [The continuous
build](https://op.oilshell.org/) tests them at every commit.

## Contributing

Oils welcomes contributions! Here's how you can get involved:

*   **Build and Test:** Start by building the "dev build" of Oils. Instructions are on the [Contributing][] page.
*   **Find Issues:** Look for "good first issue" labels on [Github issues](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
*   **Small Contributions:** Contributions are welcome, including failing [spec tests](https://oils.pub/cross-ref.html#spec-test).
*   **Python Focus:** Code in Python; semi-automated translation to C++ handles performance.
*   **Influence Design:** Share your ideas on [YSH][].

### I aim for 24 hour response time

Please feel free to ping `andychu` on Zulip or Github if you're **waiting** for
a pull request review!  (or to ask questions)

Usually I can respond in 24 hours. I might be traveling, in which case I'll
respond with something like *I hope to look at this by Tuesday*.

I might have also **missed** your Github message, so it doesn't hurt to ping
me.

Thank you for the contributions!

## Documentation and Resources

*   **Wiki:** The [Wiki](https://github.com/oils-for-unix/oils/wiki) offers detailed developer documentation.
*   **End-User Docs:** Available on each [release page](https://oils.pub/releases.html).
*   **Zulip:** Get help and ask questions on the [Oils Zulip Chat](https://oilshell.zulipchat.com/).
*   **Home Page:** The [Oils Home Page][home-page] has all the important links.

## Links

*   [Oils Home Page][home-page]
*   [OSH Documentation][OSH]
*   [YSH Documentation][YSH]
*   [Why Create a New Unix Shell?][why]
*   [Oils 2023 FAQ][faq-2023]
*   [Contributing][Contributing]

[home-page]: https://oils.pub/
[OSH]: https://oils.pub/cross-ref.html#OSH
[YSH]: https://oils.pub/cross-ref.html#YSH
[faq-2023]: https://www.oilshell.org/blog/2023/03/faq.html
[why]: https://www.oilshell.org/blog/2021/01/why-a-new-shell.html
[Contributing]: https://github.com/oils-for-unix/oils/wiki/Contributing