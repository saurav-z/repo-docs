<div align="left">

<h2>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-white-text.svg" width="250">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
  <img alt="Spack" src="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
</picture>

</h2>

<a href="https://github.com/spack/spack/actions/workflows/ci.yml"><img src="https://github.com/spack/spack/workflows/ci/badge.svg" alt="CI Status"></a>
<a href="https://github.com/spack/spack/actions/workflows/bootstrapping.yml"><img src="https://github.com/spack/spack/actions/workflows/bootstrap.yml/badge.svg" alt="Bootstrap Status"></a>
<a href="https://github.com/spack/spack/actions/workflows/build-containers.yml"><img src="https://github.com/spack/spack/actions/workflows/build-containers.yml/badge.svg" alt="Containers Status"></a>
<a href="https://spack.readthedocs.io"><img src="https://readthedocs.org/projects/spack/badge/?version=latest" alt="Documentation Status"></a>
<a href="https://codecov.io/gh/spack/spack"><img src="https://codecov.io/gh/spack/spack/branch/develop/graph/badge.svg" alt="Code coverage"/></a>
<a href="https://slack.spack.io"><img src="https://slack.spack.io/badge.svg" alt="Slack"/></a>
<a href="https://matrix.to/#/#spack-space:matrix.org"><img src="https://img.shields.io/matrix/spack-space%3Amatrix.org?label=matrix" alt="Matrix"/></a>

</div>

# Spack: A Powerful Package Manager for High-Performance Computing

Spack is a versatile package manager designed to simplify the building and installation of software across various platforms, including Linux, macOS, Windows, and supercomputers.  [**Learn more on GitHub**](https://github.com/spack/spack).

## Key Features

*   **Multi-Platform Support:** Install software on Linux, macOS, Windows, and HPC systems.
*   **Version and Configuration Management:** Easily manage multiple versions and configurations of your software.
*   **Non-Destructive Installation:** Install new package versions without breaking existing ones.
*   **Flexible Specification:** Define software versions and configurations using a simple "spec" syntax.
*   **Python-Based Package Files:** Leverage Python for creating package files, enabling single scripts for diverse builds.

## Installation

Get started with Spack by following these steps:

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

## Documentation and Resources

*   [**Full Documentation**](https://spack.readthedocs.io/)
*   `spack help` and `spack help --all` for command-line assistance.
*   `spack help --spec` for a cheat sheet on Spack syntax.
*   [**Hands-on Tutorial**](https://spack-tutorial.readthedocs.io/) - Covers basic to advanced usage, packaging, developer features, and large HPC deployments.

## Community

Join the Spack community for support, discussions, and contributions:

*   **Slack Workspace:** [spackpm.slack.com](https://spackpm.slack.com) ([Get an invitation](https://slack.spack.io))
*   **Matrix Space:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack)
*   **GitHub Discussions:** [Q&A and discussions](https://github.com/spack/spack/discussions)
*   **X:** [@spackpm](https://twitter.com/spackpm)
*   **Mailing List:** [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (for announcements)

## Contributing

Contributions are welcome!  To contribute, submit a [pull request](https://help.github.com/articles/using-pull-requests/) and adhere to the following guidelines:

*   Target the ``develop`` branch.
*   Pass Spack's tests (unit tests, documentation tests, and package build tests).
*   Be [PEP 8](https://www.python.org/dev/peps/pep-0008/) compliant.
*   Sign off all commits with `git commit --signoff`.

For further details, consult our [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html).

## Releases

For stable software installations, explore Spack's [stable releases](https://github.com/spack/spack/releases).

*   Each release has a corresponding branch (e.g., `releases/v0.14`).
*   Backported bug fixes are provided on these branches.
*   The latest release is available with the `releases/latest` tag.
*   See the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases) for more details.

## Code of Conduct

Please abide by the Spack [**Code of Conduct**](.github/CODE_OF_CONDUCT.md).

## Authors & Citation

Spack was created by Todd Gamblin, tgamblin@llnl.gov, and is maintained by a team of contributors.

If you use Spack in your publications, cite the following paper:

 * Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee,
   Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
   [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
   In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

## License

Spack is licensed under the terms of both the MIT license and the Apache License (Version 2.0). Users may choose either license.

*   [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT)
*   [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE)
*   [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT)
*   [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE)

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652
```
Key improvements and SEO considerations:

*   **Clear and Concise Hook:**  The one-sentence hook is at the beginning, making the purpose of Spack immediately clear.
*   **Keyword-Rich Headings:** Headings use relevant keywords (e.g., "Package Manager," "Multi-Platform," "Installation").
*   **Bulleted Key Features:** Easy-to-scan format for quick understanding of Spack's capabilities.
*   **Clear Call to Action:**  The "Learn more on GitHub" link at the beginning, plus links throughout the content.
*   **Comprehensive Documentation Links:**  Easy access to crucial documentation is clearly shown.
*   **Community Section:** Highlights avenues for support and contribution.
*   **Contributing Guidelines:**  Addresses how to contribute.
*   **Release Information:** Clearly explains stable releases.
*   **Code of Conduct Notice:** Important for community participation.
*   **Citation Information:** Includes a clear citation to help with referencing Spack.
*   **License Information:** Includes license.
*   **Improved Formatting:**  Uses bolding and clear spacing to enhance readability.
*   **Direct Linking:** Links the original repo.