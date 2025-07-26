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

# Spack: The Flexible Package Manager for HPC and Beyond

Spack is a powerful, multi-platform package manager designed to build and install diverse versions and configurations of software, making it ideal for High-Performance Computing (HPC) environments and beyond.  [Visit the original repository](https://github.com/spack/spack) for more information.

## Key Features

*   **Multi-Platform Support:** Works seamlessly on Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installations:** Install multiple versions of the same package without conflicts.
*   **Flexible Configuration:**  Specify versions and build options using a simple "spec" syntax.
*   **Package as Code:** Package files are written in Python, allowing for efficient management of diverse builds.
*   **Reproducible Builds:**  Ensure consistency and portability of your software environment.

## Installation

Get started with Spack in a few simple steps:

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
4.  **Install a package:**
    ```bash
    spack install zlib-ng
    ```

## Documentation and Resources

*   **[Full Documentation](https://spack.readthedocs.io/)**: Comprehensive documentation covering all aspects of Spack.
*   **`spack help` and `spack help --all`**: Command-line help for quick information.
*   **`spack help --spec`**: Cheat sheet for Spack syntax.
*   **[Tutorial](https://spack-tutorial.readthedocs.io/)**: Hands-on tutorial for basic to advanced usage.

## Community

Spack thrives on community contributions.  Join the community to ask questions, discuss ideas, and contribute:

*   **Slack Workspace:** [spackpm.slack.com](https://spackpm.slack.com) ([Get an Invitation](https://slack.spack.io))
*   **Matrix Space:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack)
*   **GitHub Discussions:** [GitHub Discussions](https://github.com/spack/spack/discussions)
*   **X (Twitter):** [@spackpm](https://twitter.com/spackpm)
*   **Mailing List:** [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (for announcements)

## Contributing

We welcome contributions!  To contribute:

1.  Submit a [Pull Request](https://help.github.com/articles/using-pull-requests/).
2.  Target the `develop` branch.
3.  Ensure your PR passes all CI tests (unit, documentation, and package build tests).
4.  Adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.
5.  Sign off all commits with `git commit --signoff`.

For contributions to the community package recipes, visit the **[spack-packages repository][Packages]**.

See the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html) for more details.

## Releases

For production environments, consider using [Spack's stable releases](https://github.com/spack/spack/releases) to ensure stability.  Each release series has a corresponding branch.

*   `releases/latest` tag for the latest release.

See the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases) for more details.

## Code of Conduct

Please adhere to the [**Code of Conduct**](.github/CODE_OF_CONDUCT.md) when participating in the Spack community.

## Authors

Spack was created by Todd Gamblin, tgamblin@llnl.gov, and is maintained by a dedicated team of [contributors](https://github.com/spack/spack/graphs/contributors).

## Citing Spack

If you are referencing Spack in a publication, please cite the following paper:

 * Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee,
   Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
   [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
   In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

Use the "Cite this repository" button on GitHub for citation formats.

## License

Spack is licensed under the MIT and Apache License (Version 2.0). Users can choose either license.
See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT),
[LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE),
[COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and
[NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652