# Oils: Upgrade Your Shell with a Modern Language and Runtime

**Oils** is a revolutionary project designed to modernize your shell experience, offering an upgrade path from bash to a more powerful and efficient language. Learn more and contribute at the original [Oils repository](https://github.com/oils-for-unix/oils)!

[![Build Status](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml/badge.svg?branch=master)](https://github.com/oils-for-unix/oils/actions/workflows/all-builds.yml)
<a href="https://gitpod.io/from-referrer/">
  <img src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod" alt="Contribute with Gitpod" />
</a>

## Key Features

*   **OSH (Oil Shell):** Runs your existing bash scripts, providing compatibility.
*   **YSH:** A new shell language designed for Python and JavaScript users.
*   **Fast Performance:** Code is written in Python for ease of modification, then translated to C++ for speed and a small footprint, with no Python dependency in the deployed executable.
*   **Active Development:** The project is actively maintained and welcomes contributions.
*   **Open Source:** Available under an open-source license, allowing for community collaboration and improvement.

## Getting Started

### For Users

If you want to **use** Oils, don't clone this repo.  Instead, visit the latest release at: <https://oils.pub/release/latest/>. For more information, refer to the [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases).

### For Developers: Contributing

1.  **Build the Dev Version:** Follow the instructions on the [Contributing](https://github.com/oils-for-unix/oils/wiki/Contributing) page to build the Oils developer version.
2.  **Report Issues:** If the build fails, post on the `#oil-dev` channel of [oilshell.zulipchat.com](https://oilshell.zulipchat.com/) or file an issue on Github.
3.  **Contribute Code:** Feel free to grab an [issue from Github](https://github.com/oils-for-unix/oils/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

#### Quick Start on Linux

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

### Dev Build vs. Release Build

Again, note that the **developer build** is **very different** from the release
tarball.  The [Contributing][] page describes this difference in detail.

The release tarballs are linked from the [home page][home-page].  (Developer
builds don't work on OS X, so use the release tarballs on OS X.)

### Important: We Accept Small Contributions!

Oils is full of [many ideas](https://oils.pub/blog/), which may be
intimidating at first.

But the bar to contribution is very low.  It's basically a medium size Python
program with many tests, and many programmers know how to change such programs.
It's great for prototyping.

- For OSH compatibility, I often merge **failing [spec
  tests](https://oils.pub/cross-ref.html#spec-test)**.  You don't even
  have to write code!  The tests alone help.  I search for related tests with
  `grep xtrace spec/*.test.sh`, where `xtrace` is a shell feature.
- You only have to make your code work **in Python**.  Plain Python programs
  are easy to modify.  The semi-automated translation to C++ is a separate
  step, although it often just works.
- You can **influence the design** of [YSH][].  If you have an itch to
  scratch, be ambitious.  For example, you might want to show us how to
  implement [nonlinear pipelines](https://github.com/oils-for-unix/oils/issues/843).

### I aim for 24 hour response time

Please feel free to ping `andychu` on Zulip or Github if you're **waiting** for
a pull request review!  (or to ask questions)

Usually I can respond in 24 hours. I might be traveling, in which case I'll
respond with something like *I hope to look at this by Tuesday*.

I might have also **missed** your Github message, so it doesn't hurt to ping
me.

Thank you for the contributions!

## Documentation

*   The [Wiki](https://github.com/oils-for-unix/oils/wiki) provides extensive developer documentation.
*   **End-User Docs:** Linked from each [release page](https://oils.pub/releases.html).
*   **Need Help?** Ask on Zulip for assistance and guidance.

## Links

*   [Oils Home Page](https://oils.pub/)
*   [OSH Documentation](https://oils.pub/cross-ref.html#OSH)
*   [YSH Documentation](https://oils.pub/cross-ref.html#YSH)
*   [Oils 2023 FAQ](https://www.oilshell.org/blog/2023/03/faq.html)
*   [Why Create a New Unix Shell?](https://www.oilshell.org/blog/2021/01/why-a-new-shell.html)
*   [Repo Overview](doc/repo-overview.md)
*   [README-index.md](README-index.md) links to docs for some subdirectories.
*   [The Oils Repo Is Different From the Tarball Releases](https://github.com/oils-for-unix/oils/wiki/The-Oils-Repo-Is-Different-From-the-Tarball-Releases)