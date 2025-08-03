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

# Spack: A Powerful Package Manager for High-Performance Computing

Spack is a versatile and open-source package manager designed to simplify the building and installation of software, especially in high-performance computing (HPC) environments.  [**Explore the Spack repository**](https://github.com/spack/spack).

## Key Features

*   **Multi-Platform Support:** Works seamlessly on Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installations:** Install multiple versions and configurations of software without breaking existing installations.
*   **Flexible Specification:**  Uses a simple "spec" syntax to specify versions and configuration options.
*   **Package Recipes in Python:** Package files are written in Python, allowing for a single script to build many different configurations.
*   **Reproducible Builds:** Ensures consistency and repeatability of software installations.

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

Comprehensive documentation is available to help you master Spack:

*   [**Full Documentation**](https://spack.readthedocs.io/)
*   Run `spack help` or `spack help --all` for command-line assistance.
*   For a cheat sheet on Spack syntax, run `spack help --spec`.

## Tutorial

Get hands-on experience with Spack through our detailed tutorial:

*   [**Hands-on Tutorial**](https://spack-tutorial.readthedocs.io/)
    This tutorial covers beginner to advanced usage, packaging, and large HPC deployments, all executable within a Docker container.

## Community

Spack thrives on community contributions and welcomes your participation:

*   **Slack workspace**: [spackpm.slack.com](https://spackpm.slack.com).
    To get an invitation, visit [slack.spack.io](https://slack.spack.io).
*   **Matrix space**: [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org):
    [bridged](https://github.com/matrix-org/matrix-appservice-slack#matrix-appservice-slack) to Slack.
*   [**Github Discussions**](https://github.com/spack/spack/discussions):
    for Q&A and discussions. Note the pinned discussions for announcements.
*   **X**: [@spackpm](https://twitter.com/spackpm). Be sure to
    `@mention` us!
*   **Mailing list**: [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack):
    only for announcements. Please use other venues for discussions.

## Contributing

Contributions are welcome!  To contribute:

1.  Send us a [pull request](https://help.github.com/articles/using-pull-requests/).
2.  Contribute to Spack's community package recipes at the **[spack-packages repository][Packages]**.
3.  When contributing to Spack itself, submit a pull request to the [spack repository](https://github.com/spack/spack).
4.  Ensure your PR:
    *   Targets the `develop` branch.
    *   Passes Spack's unit tests, documentation tests, and package build tests.
    *   Is [PEP 8](https://www.python.org/dev/peps/pep-0008/) compliant.
    *   Signs off all commits with `git commit --signoff`.

   See our [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html) for local testing and git tips.

## Releases

For stable software installations in multi-user environments, use Spack's [stable releases](https://github.com/spack/spack/releases).

*   Each release has a corresponding branch (e.g., `releases/v0.14`).
*   The latest release is always available with the `releases/latest` tag.
*   See the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases) for details.

## Code of Conduct

Please adhere to Spack's [**Code of Conduct**](.github/CODE_OF_CONDUCT.md) when participating in the community.

## Authors

Spack is a community effort.  Many thanks to our [contributors](https://github.com/spack/spack/graphs/contributors).

Spack was created by Todd Gamblin, tgamblin@llnl.gov.

### Citing Spack

If you are referencing Spack in a publication, please cite the following paper:

 * Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee,
   Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
   [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
   In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

On GitHub, you can copy this citation in APA or BibTeX format via the "Cite this repository"
button. Or, see the comments in `CITATION.cff` for the raw BibTeX.

## License

Spack is distributed under the terms of both the MIT license and the Apache License (Version 2.0).

*   See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT),
    [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE),
    [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and
    [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652