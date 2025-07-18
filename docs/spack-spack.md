<div align="left">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://cdn.rawgit.com/spack/spack/develop/share/spack/logo/spack-logo-white-text.svg" width="250">
  <source media="(prefers-color-scheme: light)" srcset="https://cdn.rawgit.com/spack/spack/develop/share/spack/logo/spack-logo-text.svg" width="250">
  <img alt="Spack" src="https://cdn.rawgit.com/spack/spack/develop/share/spack/logo/spack-logo-text.svg" width="250">
</picture>
<br>
<br clear="all">

[![CI Status](https://github.com/spack/spack/workflows/ci/badge.svg)](https://github.com/spack/spack/actions/workflows/ci.yml)
[![Bootstrap Status](https://github.com/spack/spack/workflows/bootstrap.yml/badge.svg)](https://github.com/spack/spack/actions/workflows/bootstrapping.yml)
[![Containers Status](https://github.com/spack/spack/workflows/build-containers.yml/badge.svg)](https://github.com/spack/spack/actions/workflows/build-containers.yml)
[![Documentation Status](https://readthedocs.org/projects/spack/badge/?version=latest)](https://spack.readthedocs.io)
[![Code coverage](https://codecov.io/gh/spack/spack/branch/develop/graph/badge.svg)](https://codecov.io/gh/spack/spack)
[![Slack](https://slack.spack.io/badge.svg)](https://slack.spack.io)
[![Matrix](https://img.shields.io/matrix/spack-space%3Amatrix.org?label=matrix)](https://matrix.to/#/#spack-space:matrix.org)
</div>

**[Getting Started] &nbsp; • &nbsp; [Config] &nbsp; • &nbsp; [Community] &nbsp; • &nbsp; [Contributing] &nbsp; • &nbsp; [Packaging Guide]**

[Getting Started]: https://spack.readthedocs.io/en/latest/getting_started.html
[Config]: https://spack.readthedocs.io/en/latest/configuration.html
[Community]: #community
[Contributing]: https://spack.readthedocs.io/en/latest/contribution_guide.html
[Packaging Guide]: https://spack.readthedocs.io/en/latest/packaging_guide.html

## Spack: The Multi-Platform Package Manager for High-Performance Computing

Spack is a powerful, open-source package manager designed to build and install multiple versions and configurations of software on various platforms, including Linux, macOS, Windows, and supercomputers; find the original repository at [https://github.com/spack/spack](https://github.com/spack/spack).

Key Features:

*   **Multi-Platform Support:** Works seamlessly across diverse operating systems and HPC environments.
*   **Non-Destructive Installations:** Install multiple versions of packages without conflicts.
*   **Flexible Configuration:** Specify versions and build options with a simple "spec" syntax.
*   **Python-Based Packaging:** Package files are written in pure Python, enabling a single script for different builds.
*   **Dependency Management:** Automatically handles dependencies and ensures correct software linking.

## Installation

### Prerequisites

Ensure you have Python and Git installed on your system.

### Steps

1.  Clone the Spack repository:

    ```bash
    git clone --depth=2 https://github.com/spack/spack.git
    ```

2.  Set up your environment:

    ```bash
    # For bash/zsh/sh
    . spack/share/spack/setup-env.sh

    # For tcsh/csh
    source spack/share/spack/setup-env.csh

    # For fish
    . spack/share/spack/setup-env.fish
    ```

3.  Install a package (example with zlib-ng):

    ```bash
    spack install zlib-ng
    ```

## Documentation

Comprehensive documentation is available:

*   [**Full Documentation**](https://spack.readthedocs.io/)
*   Run `spack help` or `spack help --all` for command-line help.
*   For a quick syntax reference, use `spack help --spec`.

## Tutorial

A hands-on tutorial is available:

*   [**Hands-on Tutorial**](https://spack-tutorial.readthedocs.io/)
*   Covers basic to advanced usage, packaging, and HPC deployments.
*   Exercises can be completed on your laptop using a Docker container.

## Community

Spack thrives on community contributions and welcomes your participation.

Resources:

*   **Slack Workspace:** [spackpm.slack.com](https://spackpm.slack.com) (get an invite at [slack.spack.io](https://slack.spack.io)).
*   **Matrix Space:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack).
*   **GitHub Discussions:** [https://github.com/spack/spack/discussions](https://github.com/spack/spack/discussions) for Q&A and discussions.
*   **X:** [@spackpm](https://twitter.com/spackpm) - be sure to `@mention` us!
*   **Mailing List:** [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (for announcements only).

## Contributing

Contributions are welcome! Submit pull requests to the [spack repository](https://github.com/spack/spack).

*   See the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html) for details.
*   Most contributions are to the [spack-packages repository](https://github.com/spack/spack-packages).

Your PR must:

  1. Make ``develop`` the destination branch;
  2. Pass Spack's unit tests, documentation tests, and package build tests;
  3. Be [PEP 8](https://www.python.org/dev/peps/pep-0008/) compliant;
  4. Sign off all commits with `git commit --signoff`. Signoff says that you
     agree to the [Developer Certificate of Origin](https://developercertificate.org).
     Note that this is different from [signing commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits),
     which you may also do, but it's not required.

## Releases

For stable deployments, use Spack's [stable releases](https://github.com/spack/spack/releases). Each release series has a corresponding branch (e.g., `releases/v0.14`).

*   Latest release: `releases/latest` tag.
*   See [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases) for details.

## Code of Conduct

Please adhere to the [**Code of Conduct**](.github/CODE_OF_CONDUCT.md) when participating in the Spack community.

## Authors

Thanks to all of Spack's [contributors](https://github.com/spack/spack/graphs/contributors).

Spack was created by Todd Gamblin, tgamblin@llnl.gov.

## Citing Spack

If you are referencing Spack in a publication, please cite:

*   Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee,
    Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
    [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
    In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

Get citation in APA or BibTeX format via the "Cite this repository" button on GitHub or in `CITATION.cff`.

## License

Spack is available under both the MIT and Apache License (Version 2.0). Users can choose either license.
All new contributions must be made under both licenses.

See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT), [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE), [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652
```
Key changes and improvements:

*   **SEO Optimization:**  Added a concise, keyword-rich title and a one-sentence hook that highlights the core benefit.  Used relevant keywords throughout (package manager, HPC, etc.).
*   **Clear Headings:**  Uses clear, descriptive headings for each section.
*   **Bulleted Key Features:**  Highlights key features using bullet points for easy readability.
*   **Concise Language:**  Streamlined the text for better clarity and readability.
*   **Call to Actions:**  Encourages user interaction (e.g., "Install a package", "Submit pull requests").
*   **Organized Information:** The content is logically organized, making it easier for users to find what they need.
*   **Links:**  Maintained and improved links.
*   **Contribution Details:** Added explicit requirements for pull requests.
*   **Licenses Details:** Ensured that all license details were included.
*   **GitHub-specific features:** Includes GitHub links.
*   **Added back-links:**  Included a link to the original repository.