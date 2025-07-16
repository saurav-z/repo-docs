<div align="left">

<h2>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://cdn.rawgit.com/spack/spack/develop/share/spack/logo/spack-logo-white-text.svg" width="250">
  <source media="(prefers-color-scheme: light)" srcset="https://cdn.rawgit.com/spack/spack/develop/share/spack/logo/spack-logo-text.svg" width="250">
  <img alt="Spack" src="https://cdn.rawgit.com/spack/spack/develop/share/spack/logo/spack-logo-text.svg" width="250">
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

**[Getting Started] &nbsp; • &nbsp; [Config] &nbsp; • &nbsp; [Community] &nbsp; • &nbsp; [Contributing] &nbsp; • &nbsp; [Packaging Guide]**

[Getting Started]: https://spack.readthedocs.io/en/latest/getting_started.html
[Config]: https://spack.readthedocs.io/en/latest/configuration.html
[Community]: #community
[Contributing]: https://spack.readthedocs.io/en/latest/contribution_guide.html
[Packaging Guide]: https://spack.readthedocs.io/en/latest/packaging_guide.html

</div>

# Spack: A Package Manager for High-Performance Computing (HPC)

Spack is a powerful, open-source package manager designed for building and managing software on Linux, macOS, Windows, and supercomputers.  [Learn more about Spack](https://github.com/spack/spack).

**Key Features:**

*   **Multi-Platform Support:** Build and install software across Linux, macOS, Windows, and HPC environments.
*   **Non-Destructive Installations:** Install multiple versions and configurations of a package without breaking existing installations.
*   **Flexible Spec Syntax:** Easily specify versions, dependencies, and configuration options using a simple, intuitive syntax.
*   **Package Files in Python:** Customize and extend package builds with Python scripts for maximum flexibility.
*   **Reproducible Builds:** Ensure consistent builds with version control and dependency management.

**Installation**

To get started with Spack, follow these steps:

1.  **Prerequisites:** Ensure you have Python and Git installed on your system.
2.  **Clone the Repository:**
    ```bash
    git clone --depth=2 https://github.com/spack/spack.git
    ```
3.  **Set up your environment:**  Run the setup script for your shell:
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

**Documentation**

*   [**Full Documentation**](https://spack.readthedocs.io/)
*   `spack help` and `spack help --all` for command-line assistance.
*   `spack help --spec` for a cheat sheet on Spack syntax.

**Tutorial**

*   [**Hands-on Tutorial**](https://spack-tutorial.readthedocs.io/) - Learn Spack from basic to advanced usage.

**Community**

Spack thrives on community contributions. Join the conversation and contribute!

*   **Slack:** [spackpm.slack.com](https://spackpm.slack.com) ([get an invitation](https://slack.spack.io)).
*   **Matrix:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack).
*   **GitHub Discussions:** [GitHub Discussions](https://github.com/spack/spack/discussions) for Q&A and announcements.
*   **Twitter:** [@spackpm](https://twitter.com/spackpm)
*   **Mailing List:** [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (for announcements only).

**Contributing**

Contribute to Spack by submitting [pull requests](https://help.github.com/articles/using-pull-requests/).

*   Contribute to package recipes in the [spack-packages repository](https://github.com/spack/spack-packages).
*   Contribute to Spack itself via the [spack repository](https://github.com/spack/spack).

**Releases**

For stable software installations, use Spack's [stable releases](https://github.com/spack/spack/releases).

*   Each release series has a corresponding branch (e.g., `releases/v0.14`).
*   Get fixes without package churn by using release branches and `git pull`.

**Code of Conduct**

Spack follows a [**Code of Conduct**](.github/CODE_OF_CONDUCT.md).

**Authors**

Many thanks to Spack's [contributors](https://github.com/spack/spack/graphs/contributors).

Spack was created by Todd Gamblin, tgamblin@llnl.gov.

**Citing Spack**

If you use Spack in your research, please cite the following paper:

 * Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee,
   Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
   [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
   In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

**License**

Spack is available under both the [MIT license](https://github.com/spack/spack/blob/develop/LICENSE-MIT) and the [Apache License (Version 2.0)](https://github.com/spack/spack/blob/develop/LICENSE-APACHE).