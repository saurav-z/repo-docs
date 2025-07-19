<div align="left">
<h2>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://cdn.rawgit.com/spack/spack/develop/share/spack/logo/spack-logo-white-text.svg" width="250">
  <source media="(prefers-color-scheme: light)" srcset="https://cdn.rawgit.com/spack/spack/develop/share/spack/logo/spack-logo-text.svg" width="250">
  <img alt="Spack" src="https://cdn.rawgit.com/spack/spack/develop/share/spack/logo/spack-logo-text.svg" width="250">
</picture>
</h2>

**Spack: Effortlessly Build and Manage Software on Any Platform.**

**[Getting Started] &nbsp; • &nbsp; [Config] &nbsp; • &nbsp; [Community] &nbsp; • &nbsp; [Contributing] &nbsp; • &nbsp; [Packaging Guide]**

[Getting Started]: https://spack.readthedocs.io/en/latest/getting_started.html
[Config]: https://spack.readthedocs.io/en/latest/configuration.html
[Community]: #community
[Contributing]: https://spack.readthedocs.io/en/latest/contribution_guide.html
[Packaging Guide]: https://spack.readthedocs.io/en/latest/packaging_guide.html

</div>

Spack is a powerful, open-source package manager designed to simplify the process of building and installing software across various platforms, from laptops to supercomputers. Visit the [Spack GitHub Repository](https://github.com/spack/spack) to learn more.

<br>

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, macOS, Windows, and HPC environments.
*   **Non-Destructive Installations:** Install multiple versions and configurations of software without conflicts.
*   **Flexible Configuration:** Utilize a simple "spec" syntax to define versions and build options.
*   **Package Files in Python:**  Write package definitions in Python for maximum flexibility and maintainability.
*   **Reproducible Builds:** Ensure consistent software builds across different environments.

<br>

## Installation

To get started with Spack:

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
4.  **Install a package:**
    ```bash
    spack install zlib-ng
    ```

<br>

## Documentation

*   **Comprehensive Documentation:** Explore the [full documentation](https://spack.readthedocs.io/) for detailed information.
*   **Command-Line Help:** Use `spack help` or `spack help --all` for command-line assistance.
*   **Spec Syntax Cheat Sheet:** Run `spack help --spec` to learn about Spack's syntax.

<br>

## Tutorial

*   **Hands-on Tutorial:** Master Spack with the [hands-on tutorial](https://spack-tutorial.readthedocs.io/), which covers basic to advanced usage, packaging, developer features, and large HPC deployments.

<br>

## Community

Join the Spack community for support, discussions, and contributions:

*   **Slack:** [spackpm.slack.com](https://spackpm.slack.com) (get an invitation at [slack.spack.io](https://slack.spack.io)).
*   **Matrix:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack).
*   **GitHub Discussions:** [GitHub Discussions](https://github.com/spack/spack/discussions) for Q&A and discussions.
*   **X (Twitter):** [@spackpm](https://twitter.com/spackpm) - be sure to `@mention` us!
*   **Mailing List:** [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (announcements only).

<br>

## Contributing

Contribute to Spack by submitting [pull requests](https://help.github.com/articles/using-pull-requests/).  Focus on the [spack-packages repository](https://github.com/spack/spack-packages) for community package recipes. For contributions to Spack itself, follow these guidelines:

*   Target the `develop` branch.
*   Ensure all tests pass.
*   Adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.
*   Sign off commits with `git commit --signoff`.

<br>

## Releases

For production deployments, leverage Spack's [stable releases](https://github.com/spack/spack/releases). Use the corresponding branches (e.g., `releases/v0.14`) for bug fixes without package churn. The latest release is always available with the `releases/latest` tag. See the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases) for more details.

<br>

## Code of Conduct

Review and adhere to the Spack [Code of Conduct](.github/CODE_OF_CONDUCT.md) to ensure a welcoming and respectful community.

<br>

## Authors

Spack is a community effort. Many thanks go to Spack's [contributors](https://github.com/spack/spack/graphs/contributors).

Spack was created by Todd Gamblin, tgamblin@llnl.gov.

<br>

## Citing Spack

If you are referencing Spack in a publication, please cite the following paper:

 * Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee,
   Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
   [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
   In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

On GitHub, you can copy this citation in APA or BibTeX format via the "Cite this repository" button. Or, see the comments in `CITATION.cff` for the raw BibTeX.

<br>

## License

Spack is distributed under the terms of both the MIT license and the Apache License (Version 2.0). Users may choose either license, at their option.

All new contributions must be made under both the MIT and Apache-2.0 licenses.

See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT), [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE), [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652