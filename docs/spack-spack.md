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

Spack is a powerful, multi-platform package manager designed for building and managing software on Linux, macOS, Windows, and supercomputers.  

*   **Multi-Platform Support:**  Works seamlessly across various operating systems and architectures.
*   **Non-Destructive Installations:** Install multiple versions and configurations of software without conflicts.
*   **Flexible Specification:**  Use a simple syntax to specify versions, configurations, and dependencies.
*   **Python-Based Packaging:** Package files are written in Python, allowing for a single script to manage multiple builds.
*   **Easy to Use:** Get up and running quickly with the command-line interface.

For a more detailed overview, see the [Feature Overview](https://spack.readthedocs.io/en/latest/features.html)

## Installation

To install Spack, ensure you have Python and Git installed. Then:

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

Finally, install your first package:

```bash
spack install zlib-ng
```

## Documentation

Comprehensive documentation is available at [Spack Documentation](https://spack.readthedocs.io/). You can also use `spack help` or `spack help --all` for command-line assistance and `spack help --spec` for a cheat sheet on Spack syntax.

## Tutorial

Explore the hands-on [Spack Tutorial](https://spack-tutorial.readthedocs.io/) to learn basic to advanced usage, packaging, developer features, and HPC deployments.

## Community

Spack is an open-source project. Join the community for questions, discussions, and contributions.

*   **Slack:** [spackpm.slack.com](https://spackpm.slack.com) (get an invite at [slack.spack.io](https://slack.spack.io))
*   **Matrix:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack)
*   **GitHub Discussions:** [Spack Discussions](https://github.com/spack/spack/discussions)
*   **X (Twitter):** [@spackpm](https://twitter.com/spackpm)
*   **Mailing List:** [Spack Mailing List](https://groups.google.com/d/forum/spack)

## Contributing

Contribute to Spack by submitting [pull requests](https://help.github.com/articles/using-pull-requests/) to the main [Spack repository](https://github.com/spack/spack).

To contribute to community package recipes, visit the [spack-packages repository][Packages].

Your PR must:

  1.  Target the `develop` branch.
  2.  Pass Spack's tests.
  3.  Be PEP 8 compliant.
  4.  Include `git commit --signoff` for all commits, agreeing to the [Developer Certificate of Origin](https://developercertificate.org).

See the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html) for local testing guidance.

## Releases

For stable deployments, utilize Spack's [stable releases](https://github.com/spack/spack/releases). Each release series has a corresponding branch (e.g., `releases/v0.14` for `0.14.x`). Bug fixes are backported to these branches. The latest release is available with the `releases/latest` tag.

See the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases) for more details.

## Code of Conduct

Please adhere to the [Spack Code of Conduct](.github/CODE_OF_CONDUCT.md).

## Authors

Spack was created by Todd Gamblin (tgamblin@llnl.gov) and is maintained by a large community of contributors ([Contributors](https://github.com/spack/spack/graphs/contributors)).

## Citing Spack

If you are referencing Spack in a publication, please cite the following paper:

 * Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee,
   Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
   [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
   In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

Get the citation in APA or BibTeX format via the "Cite this repository" button on GitHub.  See `CITATION.cff` for the raw BibTeX.

## License

Spack is licensed under both the MIT and Apache License (Version 2.0). See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT), [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE), [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652

[**Go back to the original repository**](https://github.com/spack/spack)