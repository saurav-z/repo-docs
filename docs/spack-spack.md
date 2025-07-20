<!--
  SPDX-License-Identifier: (Apache-2.0 OR MIT)
-->

<div align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://cdn.rawgit.com/spack/spack/develop/share/spack/logo/spack-logo-white-text.svg" width="250">
    <source media="(prefers-color-scheme: light)" srcset="https://cdn.rawgit.com/spack/spack/develop/share/spack/logo/spack-logo-text.svg" width="250">
    <img alt="Spack" src="https://cdn.rawgit.com/spack/spack/develop/share/spack/logo/spack-logo-text.svg" width="250">
  </picture>
</div>

[![CI Status](https://github.com/spack/spack/workflows/ci/badge.svg)](https://github.com/spack/spack/actions/workflows/ci.yml)
[![Bootstrap Status](https://github.com/spack/spack/workflows/bootstrap.yml/badge.svg)](https://github.com/spack/spack/actions/workflows/bootstrapping.yml)
[![Containers Status](https://github.com/spack/spack/workflows/build-containers.yml/badge.svg)](https://github.com/spack/spack/actions/workflows/build-containers.yml)
[![Documentation Status](https://readthedocs.org/projects/spack/badge/?version=latest)](https://spack.readthedocs.io)
[![Code coverage](https://codecov.io/gh/spack/spack/branch/develop/graph/badge.svg)](https://codecov.io/gh/spack/spack)
[![Slack](https://slack.spack.io/badge.svg)](https://slack.spack.io)
[![Matrix](https://img.shields.io/matrix/spack-space%3Amatrix.org?label=matrix)](https://matrix.to/#/#spack-space:matrix.org)

**[Getting Started](https://spack.readthedocs.io/en/latest/getting_started.html) &nbsp; • &nbsp; [Config](https://spack.readthedocs.io/en/latest/configuration.html) &nbsp; • &nbsp; [Community](#community) &nbsp; • &nbsp; [Contributing](https://spack.readthedocs.io/en/latest/contribution_guide.html) &nbsp; • &nbsp; [Packaging Guide](https://spack.readthedocs.io/en/latest/packaging_guide_creation.html) &nbsp; • &nbsp; [Packages](https://github.com/spack/spack-packages)**

# Spack: A Powerful Package Manager for High-Performance Computing (HPC)

Spack is a versatile package manager designed to build, install, and manage multiple versions and configurations of software, perfect for HPC environments and beyond. [Learn more about Spack](https://github.com/spack/spack).

## Key Features

*   **Multi-Platform Support:** Works seamlessly on Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installations:** Install new versions without breaking existing installations; allowing for coexistence of many configurations of the same package.
*   **Flexible Configuration:**  Uses a simple "spec" syntax to define versions and configuration options.
*   **Python-Based Package Files:** Package files are written in pure Python, allowing a single script to handle many builds.
*   **Build Software Your Way:** Build your software with the configurations you need.

See the [Feature Overview](https://spack.readthedocs.io/en/latest/features.html) for detailed examples.

## Installation

Before you start, ensure you have Python and Git installed. Then, follow these steps to install Spack:

```bash
git clone --depth=2 https://github.com/spack/spack.git
```

Next, set up your environment:

```bash
# For bash/zsh/sh
. spack/share/spack/setup-env.sh

# For tcsh/csh
source spack/share/spack/setup-env.csh

# For fish
. spack/share/spack/setup-env.fish
```

Finally, install a package:

```bash
spack install zlib-ng
```

## Documentation

Comprehensive documentation is available:

*   [**Full documentation**](https://spack.readthedocs.io/)
*   Run `spack help` or `spack help --all` for command-line assistance.
*   For a syntax cheat sheet, run `spack help --spec`.

## Tutorial

Get hands-on with Spack through our tutorial:

*   [**Hands-on tutorial**](https://spack-tutorial.readthedocs.io/)

The tutorial covers everything from the basics to advanced topics, including packaging, developer features, and HPC deployments. You can complete the exercises on your laptop using a Docker container.

## Community

Spack is an open-source project. We encourage questions, discussions, and contributions.

Resources:

*   **Slack workspace**: [spackpm.slack.com](https://spackpm.slack.com). Get an invitation at [slack.spack.io](https://slack.spack.io).
*   **Matrix space**: [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org): [bridged](https://github.com/matrix-org/matrix-appservice-slack#matrix-appservice-slack) to Slack.
*   [**Github Discussions**](https://github.com/spack/spack/discussions): For Q&A and discussions.  Check the pinned discussions for announcements.
*   **X (Twitter)**: [@spackpm](https://twitter.com/spackpm).
*   **Mailing list**: [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack): Announcements only. Use other venues for discussions.

## Contributing

Contributions are welcome!

For community package recipes, visit the [spack-packages repository](https://github.com/spack/spack-packages).

To contribute to Spack itself, submit a pull request to the [spack repository](https://github.com/spack/spack).

Your pull request must:

1.  Target the `develop` branch.
2.  Pass Spack's unit, documentation, and package build tests.
3.  Be [PEP 8](https://www.python.org/dev/peps/pep-0008/) compliant.
4.  Sign off commits with `git commit --signoff` to agree to the [Developer Certificate of Origin](https://developercertificate.org).

We enforce these guidelines through our continuous integration (CI) process.  See the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html) for local testing and Git tips.

## Releases

For stable deployments, use Spack's [stable releases](https://github.com/spack/spack/releases).

Each release series has a corresponding branch (e.g., `releases/v0.14`). We backport bug fixes to these branches.

The latest release is always tagged `releases/latest`.

See the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases) for more information.

## Code of Conduct

Please adhere to our [**Code of Conduct**](.github/CODE_OF_CONDUCT.md).

## Authors

Thanks to Spack's [contributors](https://github.com/spack/spack/graphs/contributors)!

Spack was created by Todd Gamblin, tgamblin@llnl.gov.

## Citing Spack

If you're referencing Spack in a publication, cite the following paper:

 *   Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee, Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
    [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
    In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

Use the "Cite this repository" button on GitHub for APA or BibTeX formats.  Alternatively, see the comments in `CITATION.cff` for raw BibTeX.

## License

Spack is available under the [MIT license](https://github.com/spack/spack/blob/develop/LICENSE-MIT) and the [Apache License, Version 2.0](https://github.com/spack/spack/blob/develop/LICENSE-APACHE). You can choose either license.

All new contributions must be made under both licenses.

See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT), [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE), [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652