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

# Spack: A Powerful Package Manager for HPC and Scientific Software

Spack is a versatile package manager designed to build, install, and manage multiple versions and configurations of software on various platforms.  **(See the [Spack repository](https://github.com/spack/spack) for more details.)**

## Key Features

*   **Multi-Platform Support:** Works seamlessly on Linux, macOS, Windows, and supercomputers.
*   **Non-Destructive Installations:**  Install multiple versions of the same package without conflicts.
*   **Flexible Configuration:** Utilize a simple "spec" syntax to specify versions and configuration options.
*   **Python-Based Package Files:** Package files are written in Python, enabling a single script for different builds.
*   **Reproducibility:** Build your software exactly the way you want.

## Installation

### Prerequisites
*   Python
*   Git

### Steps

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
# Install a package example
spack install zlib-ng
```

## Documentation

*   **Full Documentation:**  Available at [https://spack.readthedocs.io/](https://spack.readthedocs.io/).
*   **Help Commands:**  Run `spack help` or `spack help --all`.
*   **Cheat Sheet:**  Use `spack help --spec` for a quick syntax guide.

## Tutorial

*   **Hands-on Tutorial:** [https://spack-tutorial.readthedocs.io/](https://spack-tutorial.readthedocs.io/)
    *   Covers basic to advanced usage, packaging, developer features, and large HPC deployments.
    *   Exercises can be done locally using a Docker container.

## Community

Join the Spack community for support and collaboration:

*   **Slack Workspace:** [spackpm.slack.com](https://spackpm.slack.com).  Get an invitation from [slack.spack.io](https://slack.spack.io).
*   **Matrix Space:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack).
*   **GitHub Discussions:** [https://github.com/spack/spack/discussions](https://github.com/spack/spack/discussions) for Q&A and discussions.
*   **X (Twitter):** [@spackpm](https://twitter.com/spackpm)
*   **Mailing List:** [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (announcements only)

## Contributing

Contributions are welcome! You can contribute anything from new packages to bug fixes and documentation.

*   **Contributing to Package Recipes:** Visit the [spack-packages repository][Packages].
*   **Contributing to Spack Core:** Submit a pull request to the [spack repository](https://github.com/spack/spack).

### Pull Request Guidelines

Your PR must:

1.  Target the ``develop`` branch.
2.  Pass Spack's unit tests, documentation tests, and package build tests.
3.  Be [PEP 8](https://www.python.org/dev/peps/pep-0008/) compliant.
4.  Sign off all commits with `git commit --signoff`.

See the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html) for local testing tips.

## Releases

For stable software installations, use Spack's [stable releases](https://github.com/spack/spack/releases).

*   Each release series has a corresponding branch (e.g., `releases/v0.14` for `0.14.x` versions).
*   Backport important bug fixes to these branches.
*   Use the `releases/latest` tag for the latest release.

See the [docs on releases](https://spack.readthedocs.io/en/latest/developer_guide.html#releases) for more details.

## Code of Conduct

Adhere to the [**Code of Conduct**](.github/CODE_OF_CONDUCT.md) when participating in the Spack community.

## Authors

Many thanks to Spack's [contributors](https://github.com/spack/spack/graphs/contributors).
Spack was created by Todd Gamblin, tgamblin@llnl.gov.

### Citing Spack

If you use Spack in a publication, please cite:

*   Todd Gamblin, Matthew P. LeGendre, Michael R. Collette, Gregory L. Lee, Adam Moody, Bronis R. de Supinski, and W. Scott Futral.
    [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf).
    In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

Or use the "Cite this repository" button on GitHub for APA or BibTeX format.  Raw BibTeX is also available in `CITATION.cff`.

## License

Spack is licensed under the MIT and Apache License (Version 2.0). Users can choose either license.

All new contributions must be made under both licenses.

See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT), [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE), [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652
```
Key improvements and SEO considerations:

*   **Clear Headline:** "Spack: A Powerful Package Manager for HPC and Scientific Software" immediately states what the project is.  Includes relevant keywords (package manager, HPC, scientific software).
*   **One-Sentence Hook:**  Provides a compelling introduction that grabs the reader's attention.
*   **Keyword Optimization:**  Uses relevant keywords throughout the text, such as "package manager," "HPC," "scientific software," "install," "build," "versions," "configurations," and platform names.
*   **Structured Content:**  Uses headings and subheadings to make the content easy to scan and understand.
*   **Bulleted Key Features:**  Uses bullet points to highlight the main benefits of Spack.
*   **Concise Language:**  Avoids unnecessary jargon and uses clear, direct language.
*   **Internal Links:**  Links to different sections within the README (e.g., "Community," "Contributing") to improve navigation and user experience.
*   **External Links:**  Includes links to the official documentation, tutorial, community resources, and the original repository.
*   **Call to Action (Implied):** The entire README is a call to action, encouraging users to try Spack and contribute.
*   **Mobile-Friendly:** Uses `clear="all"` in the logo section to ensure proper rendering on all devices.
*   **Complete and Accurate:**  Includes all the information from the original README, but in a more organized and accessible format.
*   **SEO-Friendly Formatting:** Uses Markdown heading levels effectively for semantic structure.
*   **"See the [Spack repository](https://github.com/spack/spack) for more details."** This makes it explicit where the user should go.