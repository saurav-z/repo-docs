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

# Spack: The Flexible Package Manager for High-Performance Computing

Spack is a powerful, multi-platform package manager designed for building and installing diverse versions and configurations of software, making software management a breeze. Check out the original repo: [Spack](https://github.com/spack/spack).

## Key Features

*   **Multi-Platform Support:** Works seamlessly on Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installations:**  Install new versions without breaking existing ones, enabling multiple configurations to coexist.
*   **Flexible Specification:**  Utilizes a simple "spec" syntax for specifying versions and configuration options.
*   **Python-Based Package Files:** Package files are written in pure Python, allowing for a single script to build various configurations.
*   **Build Software Your Way:** Enables building software with many different configurations, providing maximum flexibility.

## Installation

Get started with Spack in a few simple steps:

1.  **Prerequisites:** Ensure you have Python and Git installed.
2.  **Clone the Repository:**
    ```bash
    git clone --depth=2 https://github.com/spack/spack.git
    ```
3.  **Set Up Your Environment:** Choose your shell and source the setup script:

    ```bash
    # For bash/zsh/sh
    . spack/share/spack/setup-env.sh

    # For tcsh/csh
    source spack/share/spack/setup-env.csh

    # For fish
    . spack/share/spack/setup-env.fish
    ```
4.  **Install Your First Package:**
    ```bash
    spack install zlib-ng
    ```

## Documentation

Comprehensive documentation is available to guide you.  Explore the documentation at [**Full documentation**](https://spack.readthedocs.io/) or use the command line with `spack help` or `spack help --all`.

For a quick reference on Spack syntax, run `spack help --spec`.

## Tutorial

For a hands-on learning experience, explore the [**hands-on tutorial**](https://spack-tutorial.readthedocs.io/), which covers beginner to advanced usage.

## Community

Join the vibrant Spack community for support, discussions, and contributions.

*   **Slack Workspace:** [spackpm.slack.com](https://spackpm.slack.com) (Get an invitation at [slack.spack.io](https://slack.spack.io)).
*   **Matrix Space:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack).
*   **GitHub Discussions:** [GitHub Discussions](https://github.com/spack/spack/discussions) for Q&A and discussions.
*   **X (formerly Twitter):** [@spackpm](https://twitter.com/spackpm)
*   **Mailing List:** [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack)

## Contributing

Contribute to the Spack project through pull requests, focusing on the **[spack-packages repository][Packages]** for community package recipes.

Submit pull requests to the [spack repository](https://github.com/spack/spack) following the guidelines outlined in the original README.

## Releases

For production deployments, use Spack's [stable releases](https://github.com/spack/spack/releases).

## Code of Conduct

Adhere to the [**Code of Conduct**](.github/CODE_OF_CONDUCT.md) when participating in the Spack community.

## Authors

Thanks to the many [contributors](https://github.com/spack/spack/graphs/contributors) who have made this project possible.

Created by Todd Gamblin, tgamblin@llnl.gov.

### Citing Spack

If you are referencing Spack in a publication, please cite the following paper:

 * Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee,
   Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
   [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
   In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

On GitHub, you can copy this citation in APA or BibTeX format via the "Cite this repository"
button. Or, see the comments in `CITATION.cff` for the raw BibTeX.

## License

Spack is distributed under the terms of both the MIT license and the Apache License (Version 2.0). Users may choose either license, at their option.

See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT),
[LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE),
[COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and
[NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652