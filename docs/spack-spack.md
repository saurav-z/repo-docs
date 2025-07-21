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
</div>

# Spack: The Multi-Platform Package Manager for HPC and Beyond

Spack is a powerful package manager designed to build and install software across various platforms, providing unparalleled flexibility and control.  [Learn more on GitHub](https://github.com/spack/spack).

**Key Features:**

*   **Multi-Platform Support:** Works seamlessly on Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installations:** Install multiple versions and configurations without breaking existing installations.
*   **Flexible "Spec" Syntax:** Specify versions and configuration options with a simple and intuitive syntax.
*   **Package Definition in Python:** Write package files in pure Python for maximum flexibility and customization.
*   **Build Software Your Way:** Build your software with the exact configurations you need.

## Installation

Before installing Spack, ensure you have Python and Git installed.  Then:

```bash
git clone --depth=2 https://github.com/spack/spack.git
```

```bash
# For bash/zsh/sh
. spack/share/spack/setup-env.sh

# For tcsh/csh
source spack/share/spack/setup-env.csh

# For fish
. spack/share/spack/setup-env.fish
```

```bash
# Now you're ready to install a package!
spack install zlib-ng
```

## Documentation

Comprehensive documentation is available:  [**Full documentation**](https://spack.readthedocs.io/)

Use these commands for quick help:

*   `spack help`
*   `spack help --all`
*   `spack help --spec` (for a cheat sheet on Spack syntax)

## Tutorial

Enhance your Spack skills with our hands-on tutorial:
[**hands-on tutorial**](https://spack-tutorial.readthedocs.io/).
It covers basic to advanced usage, packaging, developer features, and large HPC deployments.  You can do all of the exercises on your own laptop using a Docker container.

## Community

Spack thrives on community contributions. Join us!

*   **Slack workspace**: [spackpm.slack.com](https://spackpm.slack.com).  Get an invite: [slack.spack.io](https://slack.spack.io).
*   **Matrix space**: [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack).
*   [**Github Discussions**](https://github.com/spack/spack/discussions): for Q&A and discussions.
*   **X**: [@spackpm](https://twitter.com/spackpm).  `@mention` us!
*   **Mailing list**: [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (for announcements).

## Contributing

We welcome your contributions!  Submit a [pull request](https://help.github.com/articles/using-pull-requests/) to the [Spack repository](https://github.com/spack/spack).

To contribute to community package recipes, visit the **[spack-packages repository][Packages]**.

Ensure your PR meets these requirements:

1.  Destination branch: `develop`.
2.  Pass all tests (unit, documentation, and package build).
3.  Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines.
4.  Sign off commits with `git commit --signoff` (Developer Certificate of Origin).

## Releases

For stable deployments, use Spack's [stable releases](https://github.com/spack/spack/releases).
Each release series has a corresponding branch (e.g., `releases/v0.14`). The latest release is tagged as `releases/latest`. See the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases).

## Code of Conduct

Please adhere to our [**Code of Conduct**](.github/CODE_OF_CONDUCT.md).

## Authors

Many thanks to our [contributors](https://github.com/spack/spack/graphs/contributors).

Spack was created by Todd Gamblin, tgamblin@llnl.gov.

### Citing Spack

If you're using Spack in your work, please cite the following paper:

 * Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee,
   Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
   [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
   In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

(Get the citation in APA or BibTeX format via the "Cite this repository" button on GitHub or see `CITATION.cff`.)

## License

Spack is licensed under the MIT and Apache 2.0 licenses. Choose the one that suits you.
All new contributions require both licenses.

See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT),
[LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE),
[COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and
[NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652