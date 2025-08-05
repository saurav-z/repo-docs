<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-white-text.svg" width="250">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
    <img alt="Spack" src="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
  </picture>
</div>

## Spack: Effortlessly Manage Software Builds and Dependencies

[Spack](https://github.com/spack/spack) is a powerful, multi-platform package manager designed to simplify the building, installation, and management of software, making complex HPC deployments easier.

[![CI Status](https://github.com/spack/spack/workflows/ci/badge.svg)](https://github.com/spack/spack/actions/workflows/ci.yml)
[![Bootstrap Status](https://github.com/spack/spack/workflows/bootstrap.yml/badge.svg)](https://github.com/spack/spack/actions/workflows/bootstrapping.yml)
[![Containers Status](https://github.com/spack/spack/workflows/build-containers.yml/badge.svg)](https://github.com/spack/spack/actions/workflows/build-containers.yml)
[![Documentation Status](https://readthedocs.org/projects/spack/badge/?version=latest)](https://spack.readthedocs.io)
[![Code coverage](https://codecov.io/gh/spack/spack/branch/develop/graph/badge.svg)](https://codecov.io/gh/spack/spack)
[![Slack](https://slack.spack.io/badge.svg)](https://slack.spack.io)
[![Matrix](https://img.shields.io/matrix/spack-space%3Amatrix.org?label=matrix)](https://matrix.to/#/#spack-space:matrix.org)

**Key Features:**

*   **Multi-Platform Support:** Works seamlessly on Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installations:** Install multiple versions and configurations of packages without conflicts.
*   **Flexible Dependency Management:** Define dependencies with a simple "spec" syntax, specifying versions and configurations.
*   **Python-Based Packaging:** Package files are written in Python, enabling efficient and customizable build processes.
*   **Reproducible Builds:** Ensures consistent builds across different environments.

### Getting Started

1.  **Prerequisites:** Ensure you have Python and Git installed.
2.  **Clone the Repository:**
    ```bash
    git clone --depth=2 https://github.com/spack/spack.git
    ```
3.  **Set Up Your Environment:** Choose the appropriate command based on your shell:
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

### Documentation

*   [**Full Documentation**](https://spack.readthedocs.io/)
*   Run `spack help` or `spack help --all` for command-line help.
*   Get a cheat sheet with `spack help --spec`.

### Tutorial

Explore our [**hands-on tutorial**](https://spack-tutorial.readthedocs.io/) to learn basic to advanced Spack usage, packaging, and HPC deployments.

### Community

Join the Spack community and contribute to its development!

*   **Slack:** [spackpm.slack.com](https://spackpm.slack.com) ([invite](https://slack.spack.io))
*   **Matrix:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org)
*   **GitHub Discussions:** [https://github.com/spack/spack/discussions](https://github.com/spack/spack/discussions)
*   **X (Twitter):** [@spackpm](https://twitter.com/spackpm)
*   **Mailing List:** [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack)

### Contributing

Contributions are welcome!  Please submit pull requests to the [spack repository](https://github.com/spack/spack).

*   Target the `develop` branch.
*   Ensure your code passes unit tests, documentation tests, and package build tests.
*   Follow PEP 8 guidelines.
*   Sign off commits with `git commit --signoff`.

See the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html) for details.

### Releases

For stable software installations, use [Spack's stable releases](https://github.com/spack/spack/releases).  The latest release is tagged as `releases/latest`. See the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases) for more information.

### Code of Conduct

The Spack community adheres to a [**Code of Conduct**](.github/CODE_OF_CONDUCT.md).

### Authors

Many thanks to Spack's [contributors](https://github.com/spack/spack/graphs/contributors). Spack was created by Todd Gamblin (tgamblin@llnl.gov).

### Citing Spack

If you reference Spack in a publication, please cite:

*   Todd Gamblin, et al. [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf). *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

### License

Spack is available under the MIT license and the Apache License (Version 2.0). New contributions must be made under both licenses. See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT), [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE), [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)
LLNL-CODE-811652