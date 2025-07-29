<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-white-text.svg" width="250">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
    <img alt="Spack" src="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
  </picture>
  <br>
  <br clear="all">
</div>

[![CI Status](https://github.com/spack/spack/workflows/ci/badge.svg)](https://github.com/spack/spack/actions/workflows/ci.yml)
[![Bootstrap Status](https://github.com/spack/spack/workflows/bootstrap.yml/badge.svg)](https://github.com/spack/spack/actions/workflows/bootstrapping.yml)
[![Containers Status](https://github.com/spack/spack/workflows/build-containers.yml/badge.svg)](https://github.com/spack/spack/actions/workflows/build-containers.yml)
[![Documentation Status](https://readthedocs.org/projects/spack/badge/?version=latest)](https://spack.readthedocs.io)
[![Code coverage](https://codecov.io/gh/spack/spack/branch/develop/graph/badge.svg)](https://codecov.io/gh/spack/spack)
[![Slack](https://slack.spack.io/badge.svg)](https://slack.spack.io)
[![Matrix](https://img.shields.io/matrix/spack-space%3Amatrix.org?label=matrix)](https://matrix.to/#/#spack-space:matrix.org)

<br>

[**Spack's GitHub Repository**](https://github.com/spack/spack) | [Getting Started](https://spack.readthedocs.io/en/latest/getting_started.html) | [Config](https://spack.readthedocs.io/en/latest/configuration.html) | [Community](#community) | [Contributing](https://spack.readthedocs.io/en/latest/contribution_guide.html) | [Packaging Guide](https://spack.readthedocs.io/en/latest/packaging_guide_creation.html) | [Packages](https://github.com/spack/spack-packages)

## Spack: The Flexible and Powerful Package Manager for HPC and Beyond

Spack is a versatile package manager designed to build and install multiple versions and configurations of software on various platforms.

**Key Features:**

*   **Multi-Platform Support:** Works seamlessly on Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installations:** Install new package versions without breaking existing ones, allowing for flexible configurations.
*   **Spec-Driven Builds:** Utilize a simple "spec" syntax to define versions and configuration options.
*   **Python-Based Package Files:** Package files are written in Python, enabling a single script for multiple builds.
*   **Version and Configuration Control:** Build your software exactly the way you need it.

[See the Feature Overview](https://spack.readthedocs.io/en/latest/features.html) for examples and highlights.

## Installation

Before installing Spack, ensure you have Python and Git.  Then, follow these steps:

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
# Install a package!
spack install zlib-ng
```

## Documentation

Comprehensive documentation is available [here](https://spack.readthedocs.io/), or run `spack help` or `spack help --all`.  For a quick syntax reference, use `spack help --spec`.

## Tutorial

A hands-on tutorial is available [here](https://spack-tutorial.readthedocs.io/).  It covers everything from basic usage to advanced features, packaging, and large HPC deployments. Exercises can be completed on your laptop using a Docker container.

## Community

Spack thrives on community contributions.  Join us!

**Resources:**

*   **Slack workspace**: [spackpm.slack.com](https://spackpm.slack.com) (get an invitation: [slack.spack.io](https://slack.spack.io)).
*   **Matrix space**: [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack).
*   **Github Discussions**: [https://github.com/spack/spack/discussions](https://github.com/spack/spack/discussions) for Q&A and discussions.
*   **X**: [@spackpm](https://twitter.com/spackpm).
*   **Mailing list**: [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (announcements only).

## Contributing

Contributions are welcome! Submit pull requests to the [spack repository](https://github.com/spack/spack).

To contribute to Spack itself, follow these steps:

1.  Target the `develop` branch.
2.  Ensure your code passes Spack's unit tests, documentation tests, and package build tests.
3.  Adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.
4.  Sign off commits with `git commit --signoff`.

See the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html) for help with local testing.

To contribute to community package recipes, visit the [spack-packages repository](https://github.com/spack/spack-packages).

## Releases

For stable deployments, use Spack's [releases](https://github.com/spack/spack/releases).  Each release series has a corresponding branch, e.g., `releases/v0.14`.  Backports of important bug fixes are included in these branches. The latest release is tagged `releases/latest`.

[Read more on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases).

## Code of Conduct

Please adhere to the [**Code of Conduct**](.github/CODE_OF_CONDUCT.md).

## Authors

Many thanks to [Spack's contributors](https://github.com/spack/spack/graphs/contributors).

Spack was created by Todd Gamblin, tgamblin@llnl.gov.

### Citing Spack

If you're referencing Spack in your publications, cite the following paper:

*   Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee, Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
    [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
    In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

Get the BibTeX citation in the GitHub repo by clicking the "Cite this repository" button. Or see the comments in `CITATION.cff`.

## License

Spack is available under the MIT and Apache License (Version 2.0). New contributions must be licensed under both.

See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT), [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE), [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652