<div align="left">

<h2>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-white-text.svg" width="250">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
  <img alt="Spack" src="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
</picture>

<br>
<br clear="all">

<a href="https://github.com/spack/spack/actions/workflows/ci.yml"><img src="https://github.com/spack/spack/workflows/ci/badge.svg" alt="CI Status"></a>
<a href="https://github.com/spack/spack/actions/workflows/bootstrapping.yml"><img src="https://github.com/spack/spack/actions/workflows/bootstrap.yml/badge.svg" alt="Bootstrap Status"></a>
<a href="https://github.com/spack/spack/actions/workflows/build-containers.yml"><img src="https://github.com/spack/spack/actions/workflows/build-containers.yml/badge.svg" alt="Containers Status"></a>
<a href="https://spack.readthedocs.io"><img src="https://readthedocs.org/projects/spack/badge/?version=latest" alt="Documentation Status"></a>
<a href="https://codecov.io/gh/spack/spack"><img src="https://codecov.io/gh/spack/spack/branch/develop/graph/badge.svg" alt="Code coverage"/></a>
<a href="https://slack.spack.io"><img src="https://slack.spack.io/badge.svg" alt="Slack"/></a>
<a href="https://matrix.to/#/#spack-space:matrix.org"><img src="https://img.shields.io/matrix/spack-space%3Amatrix.org?label=matrix" alt="Matrix"/></a>

</h2>

</div>

**[Getting Started] &nbsp; • &nbsp; [Config] &nbsp; • &nbsp; [Community] &nbsp; • &nbsp; [Contributing] &nbsp; • &nbsp; [Packaging Guide] &nbsp; • &nbsp; [Packages]**

[Getting Started]: https://spack.readthedocs.io/en/latest/getting_started.html
[Config]: https://spack.readthedocs.io/en/latest/configuration.html
[Community]: #community
[Contributing]: https://spack.readthedocs.io/en/latest/contribution_guide.html
[Packaging Guide]: https://spack.readthedocs.io/en/latest/packaging_guide_creation.html
[Packages]: https://github.com/spack/spack-packages

## Spack: Your Ultimate Package Manager for High-Performance Computing

Spack is a powerful, multi-platform package manager that simplifies software installation and management, especially in HPC environments.  [**Explore the Spack repository on GitHub**](https://github.com/spack/spack).

### Key Features:

*   **Multi-Platform Support:** Works seamlessly on Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installations:** Install multiple versions and configurations without conflicts.
*   **Flexible Configuration:** Use a simple spec syntax to specify versions and build options.
*   **Package Definition in Python:**  Package files are written in Python, making it easy to manage diverse builds.
*   **Reproducible Builds:**  Ensure consistency across different systems and environments.

### Installation

Get started with Spack in a few simple steps:

1.  **Prerequisites:** Ensure you have Python and Git installed.
2.  **Clone the Repository:**
    ```bash
    git clone --depth=2 https://github.com/spack/spack.git
    ```
3.  **Set up your environment:**  Choose the appropriate setup script for your shell:

    ```bash
    # For bash/zsh/sh
    . spack/share/spack/setup-env.sh

    # For tcsh/csh
    source spack/share/spack/setup-env.csh

    # For fish
    . spack/share/spack/setup-env.fish
    ```
4.  **Install a Package:**

    ```bash
    spack install zlib-ng
    ```

### Documentation

*   **Comprehensive Documentation:**  Visit the [**Spack Documentation**](https://spack.readthedocs.io/) for detailed information.
*   **Command-Line Help:** Use `spack help` or `spack help --all` for quick reference.
*   **Spec Syntax Cheat Sheet:**  Run `spack help --spec` to learn about Spack's spec syntax.

### Tutorial

*   **Hands-on Tutorial:**  Learn the ropes with the [**Spack Tutorial**](https://spack-tutorial.readthedocs.io/), covering basic to advanced usage, packaging, and HPC deployments.  Run the exercises on your laptop using a Docker container.

### Community

Join the vibrant Spack community!

*   **Slack:** [spackpm.slack.com](https://spackpm.slack.com) ([Get an invite](https://slack.spack.io)).
*   **Matrix:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack).
*   **GitHub Discussions:**  [GitHub Discussions](https://github.com/spack/spack/discussions) for Q&A and discussions.
*   **X (Twitter):** [@spackpm](https://twitter.com/spackpm).
*   **Mailing List:** [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (for announcements).

### Contributing

Contributions are welcome! Submit pull requests to the [Spack repository](https://github.com/spack/spack).  See the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html) for details, including:

1.  Destination branch: `develop`
2.  Ensure CI tests pass.
3.  PEP 8 compliance
4.  Sign off commits with `git commit --signoff`.

For community package contributions, see the [spack-packages repository][Packages].

### Releases

For stable deployments, use Spack's [releases](https://github.com/spack/spack/releases). Each release has a corresponding branch (e.g., `releases/v0.14`).  The latest release is tagged as `releases/latest`.

### Code of Conduct

Spack adheres to a [**Code of Conduct**](.github/CODE_OF_CONDUCT.md).

### Authors

Many thanks to Spack's [contributors](https://github.com/spack/spack/graphs/contributors). Spack was created by Todd Gamblin, tgamblin@llnl.gov.

### Citing Spack

If you're using Spack in your research, please cite:

 * Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee,
   Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
   [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
   In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

### License

Spack is available under the terms of both the [MIT license](https://github.com/spack/spack/blob/develop/LICENSE-MIT) and the [Apache License (Version 2.0)](https://github.com/spack/spack/blob/develop/LICENSE-APACHE).

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652
```
Key improvements and SEO optimization:

*   **Clear Title:**  "Spack: Your Ultimate Package Manager for High-Performance Computing"  uses the project name and a relevant keyword ("package manager", "high-performance computing").
*   **SEO-Friendly Introduction:**  The first sentence is a strong hook, explaining what Spack *is* and its core benefit.
*   **Keyword Usage:**  The document incorporates relevant keywords naturally throughout (e.g., "package manager," "HPC," "install," "configuration," "build").
*   **Clear Headings and Structure:**  Uses headings and subheadings for easy readability and scannability.
*   **Bulleted Key Features:**  Highlights the main benefits of Spack.
*   **Call to Action:**  Encourages the reader to explore further.
*   **Concise Language:**  Avoids unnecessary jargon.
*   **Internal Linking:** The "Getting Started" and other documentation sections are linked to other sections on the README file for easy navigation.
*   **Direct Links:**  Includes direct links back to key resources.
*   **Complete and Relevant Information:**  Includes all the crucial sections from the original README, such as Contributing, Releases, Code of Conduct, Authors, Citing, and License, maintaining the original context.