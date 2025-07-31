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

# Spack: A Powerful Package Manager for HPC and Scientific Software

Spack is a versatile and open-source package manager designed to simplify the building, installation, and management of software, especially for high-performance computing (HPC) environments. **[Check out the original repo](https://github.com/spack/spack) to get started!**

**Key Features:**

*   **Multi-Platform Support:** Works seamlessly across Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installations:** Install multiple versions and configurations of software without conflicts.
*   **Flexible Specification:** Use a simple "spec" syntax to define versions and build options.
*   **Python-Based Package Files:** Write package files in pure Python for customization and control.
*   **Reproducible Builds:** Ensures consistent and reliable software installations.
*   **Dependency Management:** Automatically handles software dependencies.

## Installation

To get started with Spack, you'll need Python and Git.  Follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone --depth=2 https://github.com/spack/spack.git
    ```

2.  **Set Up Your Environment:**

    ```bash
    # For bash/zsh/sh
    . spack/share/spack/setup-env.sh

    # For tcsh/csh
    source spack/share/spack/setup-env.csh

    # For fish
    . spack/share/spack/setup-env.fish
    ```

3.  **Install a Package:**

    ```bash
    spack install zlib-ng
    ```

## Documentation

Comprehensive documentation is available to help you use Spack effectively:

*   [**Full Documentation**](https://spack.readthedocs.io/)
*   Run `spack help` or `spack help --all` for command-line assistance.
*   For a quick reference on Spack syntax, use `spack help --spec`.

## Tutorial

Explore our hands-on tutorial to learn the ins and outs of Spack, from basic usage to advanced topics:

*   [**Hands-on Tutorial**](https://spack-tutorial.readthedocs.io/)

The tutorial includes exercises for your laptop using a Docker container.

## Community

Spack thrives on community contributions.  Join the community to ask questions, participate in discussions, and contribute.

*   **Slack Workspace**: [spackpm.slack.com](https://spackpm.slack.com) ([Get an invitation](https://slack.spack.io))
*   **Matrix Space**: [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack)
*   **GitHub Discussions**: [GitHub Discussions](https://github.com/spack/spack/discussions)
*   **X (Twitter)**: [@spackpm](https://twitter.com/spackpm)
*   **Mailing List**: [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (for announcements only)

## Contributing

Contribute to Spack by submitting [pull requests](https://help.github.com/articles/using-pull-requests/).

To contribute to Spack's community package recipes, go to the [spack-packages repository][Packages].

If you are contributing to Spack itself:
1.  Target the `develop` branch.
2.  Ensure your changes pass tests (unit tests, documentation tests, and package build tests).
3.  Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.
4.  Sign off commits with `git commit --signoff` to agree to the [Developer Certificate of Origin](https://developercertificate.org).

For local testing and git tips, refer to the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html).

## Releases

For stable software installations, use Spack's [stable releases](https://github.com/spack/spack/releases). Each release series has a corresponding branch (e.g., `releases/v0.14`), where bug fixes are backported.  The latest release is tagged as `releases/latest`.

See the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases) for details.

## Code of Conduct

Adhere to the [**Code of Conduct**](.github/CODE_OF_CONDUCT.md) when participating in the Spack community.

## Authors

Thanks to Spack's many [contributors](https://github.com/spack/spack/graphs/contributors).

Spack was created by Todd Gamblin, tgamblin@llnl.gov.

### Citing Spack

If you reference Spack in a publication, please cite this paper:

 * Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee,
   Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
   [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
   In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

Citation is available in APA or BibTeX format via the "Cite this repository" button on GitHub.  See `CITATION.cff` for the raw BibTeX.

## License

Spack is licensed under the MIT and Apache License (Version 2.0).
New contributions must be licensed under both.

See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT),
[LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE),
[COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and
[NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652