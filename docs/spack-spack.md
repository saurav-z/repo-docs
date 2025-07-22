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

[Getting Started]: https://spack.readthedocs.io/en/latest/getting_started.html
[Config]: https://spack.readthedocs.io/en/latest/configuration.html
[Community]: #community
[Contributing]: https://spack.readthedocs.io/en/latest/contribution_guide.html
[Packaging Guide]: https://spack.readthedocs.io/en/latest/packaging_guide_creation.html
[Packages]: https://github.com/spack/spack-packages

</div>

# Spack: The Flexible Package Manager for HPC and Beyond

Spack is a powerful, open-source package manager designed to build, install, and manage multiple versions and configurations of software, making it ideal for High-Performance Computing (HPC) and other complex environments.  **[Explore the Spack repository on GitHub](https://github.com/spack/spack) to streamline your software management.**

**Key Features:**

*   **Multi-Platform Support:** Works seamlessly on Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installations:** Install new versions of packages without breaking existing ones, allowing for coexistence of multiple configurations.
*   **Flexible Configuration:** Specify versions and build options using Spack's simple "spec" syntax.
*   **Python-Based Packaging:**  Package files are written in Python, enabling a single script to handle various builds.
*   **Dependency Management:**  Intelligently handles complex software dependencies.
*   **Version Control:** Easily manage and switch between different software versions.
*   **Community-Driven:**  Benefit from a large and active community, constantly expanding the package library.

## Getting Started

1.  **Prerequisites:** Ensure you have Python and Git installed.
2.  **Clone the Repository:**
    ```bash
    git clone --depth=2 https://github.com/spack/spack.git
    ```
3.  **Set up your environment:**
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

## Documentation

Comprehensive documentation is available at [Spack Documentation](https://spack.readthedocs.io/), or by running `spack help` or `spack help --all`.

For a cheat sheet on Spack syntax, run `spack help --spec`.

## Tutorial

A hands-on tutorial is available at [Spack Tutorial](https://spack-tutorial.readthedocs.io/).  This tutorial covers basic to advanced usage, packaging, developer features, and large HPC deployments. You can use a Docker container to complete the exercises on your own laptop.

## Community

Spack is an open-source project and welcomes contributions.

*   **Slack:** [spackpm.slack.com](https://spackpm.slack.com) (Get an invitation: [slack.spack.io](https://slack.spack.io))
*   **Matrix:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack)
*   **GitHub Discussions:** [GitHub Discussions](https://github.com/spack/spack/discussions) (for Q&A and discussions)
*   **X (Twitter):** [@spackpm](https://twitter.com/spackpm)
*   **Mailing List:** [Google Groups](https://groups.google.com/d/forum/spack) (for announcements)

## Contributing

Contribute to Spack by submitting pull requests to the [spack repository](https://github.com/spack/spack).  Follow these guidelines:

1.  Target the `develop` branch.
2.  Ensure your code passes Spack's tests (unit, documentation, and package build tests).
3.  Adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.
4.  Sign off your commits with `git commit --signoff`.

For more details, see the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html).

Most community contributions involve updating packages in the **[spack-packages repository][Packages]**.

## Releases

For stable deployments, use Spack's [stable releases](https://github.com/spack/spack/releases).  Each release series has a corresponding branch (e.g., `releases/v0.14`).  The latest release is tagged as `releases/latest`.

See the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases) for more details.

## Code of Conduct

Spack adheres to a [Code of Conduct](.github/CODE_OF_CONDUCT.md).  By participating, you agree to abide by its rules.

## Authors

Thanks to all of Spack's [contributors](https://github.com/spack/spack/graphs/contributors).

Created by Todd Gamblin, tgamblin@llnl.gov.

### Citing Spack

If you reference Spack in a publication, cite the following paper:

 * Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee,
   Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
   [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
   In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

Use the "Cite this repository" button on GitHub or the BibTeX comments in `CITATION.cff`.

## License

Spack is distributed under the MIT and Apache License (Version 2.0).

See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT),
[LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE),
[COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and
[NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652