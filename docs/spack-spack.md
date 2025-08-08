<div align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-white-text.svg" width="250">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
    <img alt="Spack" src="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
  </picture>

  <br>
  <br clear="all">

  [![CI Status](https://github.com/spack/spack/workflows/ci/badge.svg)](https://github.com/spack/spack/actions/workflows/ci.yml)
  [![Bootstrap Status](https://github.com/spack/spack/workflows/bootstrap.yml/badge.svg)](https://github.com/spack/spack/actions/workflows/bootstrapping.yml)
  [![Containers Status](https://github.com/spack/spack/workflows/build-containers.yml/badge.svg)](https://github.com/spack/spack/actions/workflows/build-containers.yml)
  [![Documentation Status](https://readthedocs.org/projects/spack/badge/?version=latest)](https://spack.readthedocs.io)
  [![Code Coverage](https://codecov.io/gh/spack/spack/branch/develop/graph/badge.svg)](https://codecov.io/gh/spack/spack)
  [![Slack](https://slack.spack.io/badge.svg)](https://slack.spack.io)
  [![Matrix](https://img.shields.io/matrix/spack-space%3Amatrix.org?label=matrix)](https://matrix.to/#/#spack-space:matrix.org)

  <br>
  <br>

  [**Getting Started**](https://spack.readthedocs.io/en/latest/getting_started.html) &nbsp; | &nbsp; [**Config**](https://spack.readthedocs.io/en/latest/configuration.html) &nbsp; | &nbsp; [**Community**](#community) &nbsp; | &nbsp; [**Contributing**](#contributing) &nbsp; | &nbsp; [**Packaging Guide**](https://spack.readthedocs.io/en/latest/packaging_guide_creation.html) &nbsp; | &nbsp; [**Packages**](https://github.com/spack/spack-packages)
</div>

## Spack: The Ultimate Package Manager for High-Performance Computing

Spack is a powerful, open-source package manager designed to build and install multiple versions and configurations of software, particularly for High-Performance Computing (HPC) environments.  Find the original repository [here](https://github.com/spack/spack).

### Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installs:** Install new versions without breaking existing configurations, enabling flexible experimentation and management.
*   **Flexible Configuration:**  Utilize a simple "spec" syntax for specifying versions and configuration options.
*   **Python-Based Package Files:** Package files are written in Python, and specs allow package authors to write a single script for many different builds of the same package.
*   **Comprehensive Documentation:**  Extensive documentation, tutorials, and examples are readily available.

### Installation

Ensure you have Python and Git installed before proceeding.

1.  **Clone the Repository:**

    ```bash
    git clone --depth=2 https://github.com/spack/spack.git
    ```

2.  **Set up the Environment:**

    *   **Bash/Zsh/Sh:**

        ```bash
        . spack/share/spack/setup-env.sh
        ```

    *   **Tsch/Csh:**

        ```bash
        source spack/share/spack/setup-env.csh
        ```

    *   **Fish:**

        ```bash
        . spack/share/spack/setup-env.fish
        ```

3.  **Install a Package:**

    ```bash
    spack install zlib-ng
    ```

### Documentation

*   **Full Documentation:** Explore comprehensive guides at [https://spack.readthedocs.io/](https://spack.readthedocs.io/)
*   **Command-Line Help:** Utilize `spack help` or `spack help --all` for detailed information.
*   **Spec Syntax Cheat Sheet:** Access a quick reference with `spack help --spec`.

### Tutorial

*   **Hands-on Tutorial:**  Learn basic to advanced Spack usage through the interactive tutorial: [https://spack-tutorial.readthedocs.io/](https://spack-tutorial.readthedocs.io/) (Docker container compatible).

### Community

Join the Spack community for support, discussions, and contributions:

*   **Slack:** [spackpm.slack.com](https://spackpm.slack.com) ([slack.spack.io](https://slack.spack.io) for invite)
*   **Matrix:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org)
*   **GitHub Discussions:** [https://github.com/spack/spack/discussions](https://github.com/spack/spack/discussions)
*   **X (Twitter):** [@spackpm](https://twitter.com/spackpm)
*   **Mailing List:** [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (announcements only)

### Contributing

Contributions are welcome! Here's how to get started:

1.  **Create a Pull Request:** Submit your contributions via a pull request.
2.  **Target the `develop` Branch:**  Make `develop` your destination branch.
3.  **Ensure Compliance:**
    *   Pass all CI tests (unit, documentation, and package build tests).
    *   Adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.
    *   Sign off commits with `git commit --signoff` (Developer Certificate of Origin).
4.  **Contribute to Packages:** Contribute to the **[spack-packages repository](https://github.com/spack/spack-packages)** for package recipes.

For detailed instructions, consult the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html).

### Releases

For stable deployments, use Spack's [stable releases](https://github.com/spack/spack/releases). Each release has a corresponding branch (e.g., `releases/v0.14`). The latest release is available with the `releases/latest` tag.

More details on releases can be found in the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases).

### Code of Conduct

Adhere to the [**Code of Conduct**](.github/CODE_OF_CONDUCT.md) when participating in the Spack community.

### Authors

Thank you to all [contributors](https://github.com/spack/spack/graphs/contributors).

Spack was created by Todd Gamblin, tgamblin@llnl.gov.

### Citing Spack

If you're referencing Spack in a publication, cite the following paper:

*   Todd Gamblin, et al. [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf). In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

You can find citation information (APA, BibTeX) via the "Cite this repository" button on GitHub or in the `CITATION.cff` file.

### License

Spack is licensed under both the MIT and Apache License (Version 2.0). Choose either license at your discretion. All new contributions must be made under both licenses.

See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT), [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE), [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652
```
Key improvements:

*   **SEO-Optimized Title:**  Uses "Spack: The Ultimate Package Manager for High-Performance Computing" to target relevant search terms.
*   **Concise Hook:** A one-sentence summary to grab the reader's attention.
*   **Clear Headings:** Uses `###` for better readability and SEO.
*   **Bulleted Lists:**  Highlights key features and other important information in an easy-to-scan format.
*   **Internal Links:**  Links to relevant sections within the README (e.g., "Community", "Contributing")
*   **Explicit Links:**  The most critical links are made explicit and easy to click.
*   **Improved Formatting:**  More white space and a cleaner layout for better readability.
*   **Concise Language:**  Removed redundant phrases and streamlined the text.
*   **Call to action:** encourages community interaction.
*   **Clearer Instructions:** Installation instructions are more straightforward and easier to follow.
*   **More details:** Added extra information to the installation and documentation section.
*   **Emphasis on contributing and releases:** emphasized these sections more for clarity.