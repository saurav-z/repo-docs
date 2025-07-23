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

**[Getting Started] &nbsp; • &nbsp; [Config] &nbsp; • &nbsp; [Community] &nbsp; • &nbsp; [Contributing] &nbsp; • &nbsp; [Packaging Guide] &nbsp; • &nbsp; [Packages]**

[Getting Started]: https://spack.readthedocs.io/en/latest/getting_started.html
[Config]: https://spack.readthedocs.io/en/latest/configuration.html
[Community]: #community
[Contributing]: https://spack.readthedocs.io/en/latest/contribution_guide.html
[Packaging Guide]: https://spack.readthedocs.io/en/latest/packaging_guide_creation.html
[Packages]: https://github.com/spack/spack-packages

</div>

# Spack: The Versatile Package Manager for HPC and Beyond

Spack is a powerful, open-source package manager designed for building and managing software on a variety of platforms, perfect for HPC environments and general software development.  [Visit the original repository](https://github.com/spack/spack).

## Key Features

*   **Multi-Platform Support:** Works seamlessly on Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installations:** Install multiple versions and configurations without conflicts.
*   **Flexible Configuration:** Use a simple "spec" syntax to define versions and build options.
*   **Python-Based Package Files:** Package files are written in Python, allowing for flexible and maintainable package definitions.
*   **Dependency Management:** Handles complex dependencies with ease, ensuring compatibility.
*   **Reproducible Builds:**  Build your software consistently, regardless of the environment.
*   **Integration with HPC Systems:** Optimized for High-Performance Computing (HPC) environments.

## Installation

To get started with Spack, follow these steps:

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

*   **Comprehensive Documentation:** Access the full documentation at [https://spack.readthedocs.io/](https://spack.readthedocs.io/).
*   **Command-Line Help:** Utilize `spack help` or `spack help --all` for detailed information.
*   **Cheat Sheet:** View a Spack syntax cheat sheet with `spack help --spec`.

## Tutorial

*   **Hands-on Tutorial:**  Explore the [hands-on tutorial](https://spack-tutorial.readthedocs.io/) covering basic to advanced usage, packaging, developer features, and HPC deployments.

## Community

Spack thrives on community involvement. Join us!

*   **Slack:** Join the Slack workspace at [spackpm.slack.com](https://spackpm.slack.com) (get an invitation at [slack.spack.io](https://slack.spack.io)).
*   **Matrix:** Connect on the Matrix space at [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org).
*   **GitHub Discussions:**  Engage in Q&A and discussions at [https://github.com/spack/spack/discussions](https://github.com/spack/spack/discussions).
*   **Twitter:** Follow us on X (formerly Twitter) at [@spackpm](https://twitter.com/spackpm).
*   **Mailing List:** Subscribe to the mailing list at [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (for announcements only).

## Contributing

Contributions are welcome!  Submit pull requests via [https://help.github.com/articles/using-pull-requests/](https://help.github.com/articles/using-pull-requests/).

For community package contributions, visit the [spack-packages repository][Packages]. For core Spack contributions, submit a pull request to the [spack repository](https://github.com/spack/spack).  Ensure your PR follows these guidelines:

1.  Destination branch: `develop`
2.  Pass all tests (unit, documentation, and package build).
3.  PEP 8 compliance.
4.  Sign commits with `git commit --signoff`.

Refer to the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html) for more information.

## Releases

For stable installations, utilize Spack's [stable releases](https://github.com/spack/spack/releases).
Each release series has a corresponding branch. Bug fixes are backported to release branches.

*   **Latest Release:** The `releases/latest` tag always points to the newest release.
*   **Documentation on Releases:** More details can be found in the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases).

## Code of Conduct

Adhere to the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## Authors

Spack was created by Todd Gamblin, tgamblin@llnl.gov.
Many thanks to Spack's [contributors](https://github.com/spack/spack/graphs/contributors).

### Citing Spack

If referencing Spack in a publication, cite this paper:

*   Todd Gamblin, et al. ["The Spack Package Manager: Bringing Order to HPC Software Chaos"](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf) (SC’15, 2015).

Use the "Cite this repository" button on GitHub (APA or BibTeX format) or the BibTeX comments in `CITATION.cff`.

## License

Spack is licensed under the MIT and Apache License (Version 2.0). Choose either license. All new contributions must be made under both licenses.

*   [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT)
*   [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE)
*   [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT)
*   [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE)

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652