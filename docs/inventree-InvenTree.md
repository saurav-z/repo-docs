<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
  <h1>InvenTree</h1>
</div>

## Open-Source Inventory Management: Track and Manage Your Parts with InvenTree

InvenTree is a powerful, open-source inventory management system designed to streamline stock control and part tracking, offering a web-based admin interface and REST API for efficient management.  ([View on GitHub](https://github.com/inventree/InvenTree))

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/inventree/inventree)]
[![CI](https://github.com/inventree/InvenTree/actions/workflows/qc_checks.yaml/badge.svg)]
[![Documentation Status](https://readthedocs.org/projects/inventree/badge/?version=latest)](https://inventree.readthedocs.io/en/latest/?badge=latest)
[![Docker Build](https://github.com/inventree/InvenTree/actions/workflows/docker.yaml/badge.svg)]
[![Netlify Status](https://api.netlify.com/api/v1/badges/9bbb2101-0a4d-41e7-ad56-b63fb6053094/deploy-status)](https://app.netlify.com/sites/inventree/deploys)
[![Performance Testing](https://dev.azure.com/InvenTree/InvenTree%20test%20statistics/_apis/build/status%2Fmatmair.InvenTree?branchName=testing)](https://dev.azure.com/InvenTree/InvenTree%20test%20statistics/_build/latest?definitionId=3&branchName=testing)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/7179/badge)](https://bestpractices.coreinfrastructure.org/projects/7179)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/inventree/InvenTree/badge)](https://securityscorecards.dev/viewer/?uri=github.com/inventree/InvenTree)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=inventree_InvenTree&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=inventree_InvenTree)
[![codecov](https://codecov.io/gh/inventree/InvenTree/graph/badge.svg?token=9DZRGUUV7B)](https://codecov.io/gh/inventree/InvenTree)
[![Crowdin](https://badges.crowdin.net/inventree/localized.svg)](https://crowdin.com/project/inventree)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/inventree/inventree)
[![Docker Pulls](https://img.shields.io/docker/pulls/inventree/inventree)](https://hub.docker.com/r/inventree/inventree)
[![GitHub Org's stars](https://img.shields.io/github/stars/inventree?style=social)](https://github.com/inventree/InvenTree/)
[![Twitter Follow](https://img.shields.io/twitter/follow/inventreedb?style=social)](https://twitter.com/inventreedb)
[![Subreddit subscribers](https://img.shields.io/reddit/subreddit-subscribers/inventree?style=social)](https://www.reddit.com/r/InvenTree/)
[![Mastdon](https://img.shields.io/badge/dynamic/json?label=Mastodon&query=followers_count&url=https%3A%2F%2Fchaos.social%2Fapi%2Fv1%2Faccounts%2Flookup%3Facct=InvenTree&logo=mastodon&style=social)](https://chaos.social/@InvenTree)

<h4>
    <a href="https://demo.inventree.org/">View Demo</a>
  <span> · </span>
    <a href="https://docs.inventree.org/en/latest/">Documentation</a>
  <span> · </span>
    <a href="https://github.com/inventree/InvenTree/issues/new?template=bug_report.md&title=[BUG]">Report Bug</a>
  <span> · </span>
    <a href="https://github.com/inventree/InvenTree/issues/new?template=feature_request.md&title=[FR]">Request Feature</a>
  </h4>
</div>

## Key Features

*   **Web-Based Interface:** Access and manage your inventory through an intuitive web interface.
*   **REST API:** Integrate with external applications and automate your workflow.
*   **Part Tracking:**  Detailed tracking of parts, including specifications, and lifecycle.
*   **Stock Control:**  Manage stock levels, locations, and movements.
*   **Plugin System:** Extend functionality with custom applications and extensions.
*   **Mobile App:** Companion mobile app for convenient access on the go.

## About InvenTree

InvenTree is a robust, open-source inventory management system designed to empower businesses and individuals with comprehensive control over their parts and stock. Built with a Python/Django backend, InvenTree offers a user-friendly web interface for easy management, alongside a powerful REST API and a plugin system for extensive customization and integration. Learn more at [our website](https://inventree.org).

## Roadmap & Development

Stay informed about the latest developments and future plans:

*   **Roadmap:** Explore what's in the works via the [roadmap tag](https://github.com/inventree/InvenTree/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap)
*   **Horizon Milestone:** Check out the [horizon milestone](https://github.com/inventree/InvenTree/milestone/42) for upcoming features.

## Integration & Extensibility

InvenTree's flexible architecture supports seamless integration with other tools and systems:

*   [InvenTree API](https://docs.inventree.org/en/latest/api/)
*   [Python module](https://docs.inventree.org/en/latest/api/python/)
*   [Plugin interface](https://docs.inventree.org/en/latest/plugins/)
*   [Third party tools](https://docs.inventree.org/en/latest/plugins/integrate/)

## Technology Stack

**Server:**

*   Python
*   Django
*   DRF
*   Django Q
*   Django-Allauth

**Database:**

*   PostgreSQL
*   MySQL
*   SQLite
*   Redis

**Client:**

*   React
*   Lingui
*   React Router
*   TanStack Query
*   Zustand
*   Mantine
*   Mantine Data Table
*   CodeMirror

**DevOps:**

*   Docker
*   Crowdin
*   Codecov
*   SonarCloud
*   Packager.io

## Getting Started

Easily deploy InvenTree with the following options:

<div align="center"><h4>
    <a href="https://docs.inventree.org/en/latest/start/docker/">Docker</a>
    <span> · </span>
    <a href="https://inventree.org/digitalocean"><img src="https://www.deploytodo.com/do-btn-blue-ghost.svg" alt="Deploy to DO" width="auto" height="40" /></a>
    <span> · </span>
    <a href="https://docs.inventree.org/en/latest/start/install/">Bare Metal</a>
</h4></div>

Quick installation using the install script:

```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

For comprehensive setup instructions, consult the [getting started guide](https://docs.inventree.org/en/latest/start/install/).

## Mobile App

Access InvenTree on the go with the companion mobile app:

<div align="center"><h4>
    <a href="https://play.google.com/store/apps/details?id=inventree.inventree_app">Android Play Store</a>
     <span> · </span>
    <a href="https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone">Apple App Store</a>
</h4></div>

## Security and Community

*   **Code of Conduct:**  We are committed to providing a safe and welcoming environment. Read our [Code of Conduct](CODE_OF_CONDUCT.md).
*   **Security Policy:**  InvenTree adheres to industry best practices.  See our [Security Policy](SECURITY.md) and dedicated security pages on [our documentation site](https://docs.inventree.org/en/latest/security/).

## Contribute

Your contributions are invaluable.  Help us make InvenTree even better! Visit the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/) to get started.

## Translation

Help translate the InvenTree web application into your native language through [Crowdin](https://crowdin.com/project/inventree). Your contributions are highly appreciated.

## Support and Sponsorship

*   **Sponsor the project:**  Support InvenTree's development by [sponsoring the project](https://github.com/sponsors/inventree).
*   **Acknowledgements:**  Special thanks to [PartKeepr](https://github.com/partkeepr/PartKeepr) for inspiration. Find a list of third-party libraries in the license information dialog of your instance.

<p>This project is supported by the following sponsors:</p>
<!-- Sponsors Section -->
<p align="center">
  <!-- Sponsor images here -->
</p>

<p>With ongoing resources provided by:</p>
<!-- Supporters Section -->
<p align="center">
  <!-- Supporter images here -->
</p>

## License

InvenTree is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License.  See [LICENSE.txt](https://github.com/inventree/InvenTree/blob/master/LICENSE) for details.