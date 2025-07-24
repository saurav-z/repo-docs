<div align="left">

## Spack: Build and Manage Software Across Platforms with Ease

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-white-text.svg" width="250">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
  <img alt="Spack" src="https://raw.githubusercontent.com/spack/spack/refs/heads/develop/share/spack/logo/spack-logo-text.svg" width="250">
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

**Spack is a powerful package manager designed to simplify building and managing software across various platforms, including Linux, macOS, Windows, and supercomputers.**  Spack allows you to build software *all* the ways you want to, easily handling multiple versions and configurations without conflicts.

Key Features:

*   **Multi-Platform Support:** Works seamlessly on Linux, macOS, Windows, and HPC systems.
*   **Non-Destructive Installs:** Install multiple versions and configurations without breaking existing setups.
*   **Flexible Configuration:** Uses a simple "spec" syntax for specifying versions and build options.
*   **Package Definition in Python:** Package files are written in Python, enabling a single script for diverse builds.
*   **Open Source:**  Benefit from a vibrant and welcoming community.

[**Explore the full documentation and get started with Spack!**](https://spack.readthedocs.io/)

### Installation

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

4.  **Install your first package:**

```bash
spack install zlib-ng
```

###  Documentation

*   **[Full Documentation](https://spack.readthedocs.io/)**: Comprehensive guide to using Spack.
*   `spack help`:  Get help information within the command line.
*   `spack help --all`: Get detailed information about Spack's features.
*   `spack help --spec`: Learn about Spack syntax.

###  Tutorial

*   **[Hands-on Tutorial](https://spack-tutorial.readthedocs.io/)**:  From basic to advanced usage.  Includes exercises you can do with a Docker container.

### Community

Spack thrives on community contributions and engagement. Join us!

*   **Slack:** [spackpm.slack.com](https://spackpm.slack.com) (get an invite at [slack.spack.io](https://slack.spack.io)).
*   **Matrix:** [#spack-space:matrix.org](https://matrix.to/#/#spack-space:matrix.org) (bridged to Slack).
*   **GitHub Discussions:** [https://github.com/spack/spack/discussions](https://github.com/spack/spack/discussions) for Q&A and discussions.
*   **X (Twitter):** [@spackpm](https://twitter.com/spackpm)
*   **Mailing List:** [groups.google.com/d/forum/spack](https://groups.google.com/d/forum/spack) (for announcements only).

### Contributing

We welcome your contributions! See the [Contribution Guide](https://spack.readthedocs.io/en/latest/contribution_guide.html) for details.  To contribute:

1.  Submit a [pull request](https://help.github.com/articles/using-pull-requests/).
2.  Target the `develop` branch.
3.  Ensure your PR passes CI tests.
4.  Adhere to PEP 8.
5.  Sign off commits with `git commit --signoff`.

Contribute to Spack packages by visiting the [spack-packages repository](https://github.com/spack/spack-packages).

### Releases

For stable deployments, use Spack's [stable releases](https://github.com/spack/spack/releases).  The latest release is available with the `releases/latest` tag.

### Code of Conduct

Please adhere to the [**Code of Conduct**](.github/CODE_OF_CONDUCT.md) when participating in the Spack community.

### Authors

Many thanks to Spack's [contributors](https://github.com/spack/spack/graphs/contributors).

Spack was created by Todd Gamblin, tgamblin@llnl.gov.

### Citing Spack

If you use Spack in your research, please cite the following paper:

*   Todd Gamblin, et al. [**The Spack Package Manager: Bringing Order to HPC Software Chaos**](https://www.computer.org/csdl/proceedings/sc/2015/3723/00/2807623.pdf). In *Supercomputing 2015 (SC’15)*, Austin, Texas, November 15-20 2015. LLNL-CONF-669890.

You can copy the citation in APA or BibTeX format via the "Cite this repository" button on GitHub, or see the comments in `CITATION.cff`.

### License

Spack is licensed under the MIT and Apache License (Version 2.0).

See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT), [LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE), [COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and [NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652

[**View the original Spack repository on GitHub**](https://github.com/spack/spack)
```

Key improvements and SEO considerations:

*   **Concise and Engaging Hook:** The one-sentence hook at the beginning immediately tells the user what Spack is and its core benefit.
*   **Clear Headings:**  Organizes the content logically with relevant headings (Installation, Documentation, etc.).
*   **Keyword Optimization:** Includes relevant keywords such as "package manager," "build," "software," "platforms," "Linux," "macOS," "Windows," "HPC," etc.
*   **Bulleted Key Features:**  Highlights the main advantages of Spack in an easily digestible format.
*   **Actionable Installation Instructions:** Provides clear, step-by-step installation guidance.
*   **Internal Linking:** Links to different sections within the document.
*   **Strong Calls to Action:** Encourages users to explore the documentation and community.
*   **Community Emphasis:** Reinforces the open-source nature and welcomes contributions.
*   **SEO-Friendly Title:**  The title is optimized for search engines.
*   **Clear Organization:**  Uses Markdown for readability.
*   **Repository Link:** Includes a link back to the original GitHub repository.
*   **Citation Instructions:** Provides clear instructions for citing the project in publications.
*   **License Information:**  Clearly states the licensing terms.