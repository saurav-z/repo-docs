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

# Spack: A Powerful Package Manager for HPC and Beyond

Spack is a versatile package manager designed to build and install multiple versions and configurations of software across various platforms, making software management easier than ever.

**[Visit the official Spack repository on GitHub](https://github.com/spack/spack)**

**Key Features:**

*   **Multi-Platform Support:** Works seamlessly on Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installations:** Install new package versions without disrupting existing setups; enabling many configurations of the same package to coexist.
*   **Flexible Configuration:** Uses a simple "spec" syntax for specifying versions and build options.
*   **Python-Based Package Files:** Package definitions are written in Python, enabling a single script to handle numerous builds.
*   **Version Management:** Easily manage and switch between different versions of your software.
*   **Reproducible Builds:** Create consistent and reproducible software environments.

## Installation

Ensure you have Python and Git installed. Then, clone the Spack repository:

```bash
git clone --depth=2 https://github.com/spack/spack.git
```

Set up your environment:

```bash
# For bash/zsh/sh
. spack/share/spack/setup-env.sh

# For tcsh/csh
source spack/share/spack/setup-env.csh

# For fish
. spack/share/spack/setup-env.fish
```

Install a package:

```bash
spack install zlib-ng
```

## Documentation and Tutorials

*   **Comprehensive Documentation:**  Explore the [**Full documentation**](https://spack.readthedocs.io/) or use `spack help` or `spack help --all`.
*   **Cheat Sheet:** Get a quick syntax guide with `spack help --spec`.
*   **Hands-on Tutorial:**  A [**hands-on tutorial**](https://spack-tutorial.readthedocs.io/) covering basic to advanced usage, packaging, and HPC deployments.

## Community

Join the Spack community and contribute to the project!

*   **Slack:** [spackpm.slack.com](https://spackpm.slack.com) ([invite](https://slack.spack.io))
*   **Matrix:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org)
*   **GitHub Discussions:** [GitHub Discussions](https://github.com/spack/spack/discussions)
*   **X:** [@spackpm](https://twitter.com/spackpm)
*   **Mailing List:** [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (announcements only)

## Contributing

Contribute by submitting [pull requests](https://help.github.com/articles/using-pull-requests/). For community package recipes, visit the **[spack-packages repository][Packages]**. For contributions to Spack itself, submit a pull request to the [spack repository](https://github.com/spack/spack) with the following guidelines:

1.  Target the `develop` branch.
2.  Pass unit tests, documentation tests, and package build tests.
3.  Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.
4.  Sign off commits with `git commit --signoff`.

See the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html) for details.

## Releases

For stable deployments, use Spack's [stable releases](https://github.com/spack/spack/releases). The latest release is available with the `releases/latest` tag.

## Code of Conduct

Adhere to the Spack [**Code of Conduct**](.github/CODE_OF_CONDUCT.md) when participating in the community.

## Authors

Spack was created by Todd Gamblin (tgamblin@llnl.gov) with contributions from many others.  See [contributors](https://github.com/spack/spack/graphs/contributors).

## Citing Spack

If you use Spack in a publication, please cite:

 * Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee,
   Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
   [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
   In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

## License

Spack is available under both the MIT and Apache License (Version 2.0). See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT), [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE), [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652