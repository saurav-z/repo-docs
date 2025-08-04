<!--
  SPDX-License-Identifier: (Apache-2.0 OR MIT)
  LLNL-CODE-811652
-->

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-white-text.svg" width="250">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
    <img alt="Spack" src="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
  </picture>
</div>

<div align="center">
  <a href="https://github.com/spack/spack/actions/workflows/ci.yml"><img src="https://github.com/spack/spack/workflows/ci/badge.svg" alt="CI Status"></a>
  <a href="https://github.com/spack/spack/actions/workflows/bootstrapping.yml"><img src="https://github.com/spack/spack/actions/workflows/bootstrap.yml/badge.svg" alt="Bootstrap Status"></a>
  <a href="https://github.com/spack/spack/actions/workflows/build-containers.yml"><img src="https://github.com/spack/spack/actions/workflows/build-containers.yml/badge.svg" alt="Containers Status"></a>
  <a href="https://spack.readthedocs.io"><img src="https://readthedocs.org/projects/spack/badge/?version=latest" alt="Documentation Status"></a>
  <a href="https://codecov.io/gh/spack/spack"><img src="https://codecov.io/gh/spack/spack/branch/develop/graph/badge.svg" alt="Code coverage"/></a>
  <a href="https://slack.spack.io"><img src="https://slack.spack.io/badge.svg" alt="Slack"/></a>
  <a href="https://matrix.to/#/#spack-space:matrix.org"><img src="https://img.shields.io/matrix/spack-space%3Amatrix.org?label=matrix" alt="Matrix"/></a>
</div>

## Spack: The Flexible Package Manager for HPC

Spack is a powerful, multi-platform package manager designed for building and managing software on Linux, macOS, Windows, and supercomputers. [Visit the original repository](https://github.com/spack/spack) to learn more.

**Key Features:**

*   **Multi-Platform Support:** Works seamlessly across various operating systems and HPC environments.
*   **Non-Destructive Installations:** Install multiple versions and configurations of packages without conflicts.
*   **Flexible Specification:** Use a simple "spec" syntax to define package versions and configurations.
*   **Pure Python Packaging:** Package files are written in Python, allowing for highly customizable builds.
*   **Reproducible Builds:** Easily rebuild software with the same configurations and dependencies.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Installation](#installation)
*   [Documentation](#documentation)
*   [Tutorial](#tutorial)
*   [Community](#community)
*   [Contributing](#contributing)
*   [Releases](#releases)
*   [Code of Conduct](#code-of-conduct)
*   [Citing Spack](#citing-spack)
*   [License](#license)

## Getting Started

Explore the [Getting Started Guide](https://spack.readthedocs.io/en/latest/getting_started.html) to begin your journey with Spack.

## Installation

Before installing Spack, ensure you have Python and Git installed.

```bash
git clone --depth=2 https://github.com/spack/spack.git
```

Then, set up your environment:

```bash
# For bash/zsh/sh
. spack/share/spack/setup-env.sh

# For tcsh/csh
source spack/share/spack/setup-env.csh

# For fish
. spack/share/spack/setup-env.fish
```

Finally, install your first package:

```bash
spack install zlib-ng
```

## Documentation

Comprehensive documentation is available. Explore the [full documentation](https://spack.readthedocs.io/) or use `spack help` and `spack help --all` within the tool.

For a cheat sheet on Spack syntax, run `spack help --spec`.

## Tutorial

The [hands-on tutorial](https://spack-tutorial.readthedocs.io/) covers basic to advanced usage, packaging, and HPC deployments. Exercises can be completed on your laptop using a Docker container.

## Community

Spack thrives on community contributions. Join the conversation and contribute!

*   **Slack:** [spackpm.slack.com](https://spackpm.slack.com) ([Invitation](https://slack.spack.io)).
*   **Matrix:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack).
*   **GitHub Discussions:** [spack/spack/discussions](https://github.com/spack/spack/discussions) (Q&A and announcements).
*   **X (Twitter):** [@spackpm](https://twitter.com/spackpm)
*   **Mailing List:** [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (announcements only).

## Contributing

Contributions are welcome! Submit a [pull request](https://help.github.com/articles/using-pull-requests/) to the [spack repository](https://github.com/spack/spack).

To contribute to Spack's community package recipes, visit the [spack-packages repository][Packages] (where Packages is a link to https://github.com/spack/spack-packages).

Ensure your PR:

1.  Targets the `develop` branch.
2.  Passes Spack's tests.
3.  Is [PEP 8](https://www.python.org/dev/peps/pep-0008/) compliant.
4.  Includes signed commits with `git commit --signoff`.

See the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html) for details.

## Releases

For stable software installations, use Spack's [stable releases](https://github.com/spack/spack/releases).

Each release series has a corresponding branch (e.g., `releases/v0.14`). Backports of bug fixes are made to these branches.

The latest release is available with the `releases/latest` tag.

More details are available in the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases).

## Code of Conduct

Adhere to the [**Code of Conduct**](.github/CODE_OF_CONDUCT.md) when participating in the Spack community.

## Citing Spack

If referencing Spack in a publication, cite the following:

*   Todd Gamblin, et al. [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf). *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

Get citation formats (APA or BibTeX) via the "Cite this repository" button on GitHub or see the comments in `CITATION.cff`.

## License

Spack is licensed under both the MIT and Apache License (Version 2.0). Users can choose either license.

All new contributions must be made under both the MIT and Apache-2.0 licenses.

See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT), [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE), [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.