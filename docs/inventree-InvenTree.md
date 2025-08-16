<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
  <h1>InvenTree: Open Source Inventory Management System</h1>
</div>

<!-- Badges -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/inventree/inventree)]
[![CI](https://github.com/inventree/InvenTree/actions/workflows/qc_checks.yaml/badge.svg)]
[![Documentation Status](https://readthedocs.org/projects/inventree/badge/?version=latest)]
[![Docker Build](https://github.com/inventree/InvenTree/actions/workflows/docker.yaml/badge.svg)]
[![Netlify Status](https://api.netlify.com/api/v1/badges/9bbb2101-0a4d-41e7-ad56-b63fb6053094/deploy-status)]
[![Performance Testing](https://dev.azure.com/InvenTree/InvenTree%20test%20statistics/_apis/build/status%2Fmatmair.InvenTree?branchName=testing)]
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/7179/badge)]
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/inventree/InvenTree/badge)]
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=inventree_InvenTree&metric=sqale_rating)]
[![codecov](https://codecov.io/gh/inventree/InvenTree/graph/badge.svg?token=9DZRGUUV7B)]
[![Crowdin](https://badges.crowdin.net/inventree/localized.svg)]
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/inventree/inventree)]
[![Docker Pulls](https://img.shields.io/docker/pulls/inventree/inventree)]
[![GitHub Org's stars](https://img.shields.io/github/stars/inventree?style=social)]
[![Twitter Follow](https://img.shields.io/twitter/follow/inventreedb?style=social)]
[![Subreddit subscribers](https://img.shields.io/reddit/subreddit-subscribers/inventree?style=social)]
[![Mastdon](https://img.shields.io/badge/dynamic/json?label=Mastodon&query=followers_count&url=https%3A%2F%2Fchaos.social%2Fapi%2Fv1%2Faccounts%2Flookup%3Facct=InvenTree&logo=mastodon&style=social)]

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

## :mag: About InvenTree - Open Source Inventory Management

**InvenTree is a powerful and versatile open-source inventory management system designed to help you track parts and manage your stock with ease.**  From small businesses to large enterprises, InvenTree provides robust features for comprehensive inventory control.  This project is available on [GitHub](https://github.com/inventree/InvenTree).

**Key Features:**

*   📦 **Part Tracking:**  Detailed tracking of parts with comprehensive information.
*   🏭 **Stock Control:**  Low-level stock control and management.
*   🌐 **Web-Based Interface:**  User-friendly admin interface accessible via web browser.
*   ⚙️ **REST API:** Powerful API for integration with other systems.
*   🔌 **Plugin System:** Extensible through a plugin system for custom functionality.
*   📱 **Mobile App:** Companion mobile app for on-the-go inventory access.
*   🔄 **Integration:** Offers various integration options including API, Python Module, Plugin Interface and third party tools.

### :rocket: Get Started

Explore the [InvenTree documentation](https://docs.inventree.org/en/latest/) to get started.

#### Deployment Options:
*   **Docker:** Deploy InvenTree using Docker. [Docker Documentation](https://docs.inventree.org/en/latest/start/docker/)
*   **Digital Ocean:** Deploy InvenTree using Digital Ocean. [Digital Ocean](https://inventree.org/digitalocean)
*   **Bare Metal:** Install and run InvenTree on your server. [Bare Metal Documentation](https://docs.inventree.org/en/latest/start/install/)

#### Single Line Install:
```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

### :iphone: Mobile App

Access InvenTree on the go with the companion mobile app:

*   [Android Play Store](https://play.google.com/store/apps/details?id=inventree.inventree_app)
*   [Apple App Store](https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone)

### :bulb: Tech Stack

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

### :compass: Roadmap & Contributing

*   Explore the [Roadmap](https://github.com/inventree/InvenTree/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap) to see what's planned.
*   [Contribute](https://docs.inventree.org/en/latest/develop/contributing/) to help improve InvenTree.
*   Join the [translation efforts](https://crowdin.com/project/inventree) to help localize the application.

### :handshake: Community

The InvenTree project welcomes contributions and community support.

### :money_with_wings: Sponsor

If you find InvenTree valuable, please consider supporting the project on [GitHub Sponsors](https://github.com/sponsors/inventree).

### :lock: Security

The InvenTree project team is committed to providing a safe and welcoming environment for all users. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for more information.
Refer to the [security policy](SECURITY.md) and [documentation](https://docs.inventree.org/en/latest/security/) for more details.

### :gem: Acknowledgements

We want to acknowledge [PartKeepr](https://github.com/partkeepr/PartKeepr) as a valuable predecessor and inspiration.

### :heart: Support

<p>This project is supported by the following sponsors:</p>

<p align="center">
<a href="https://github.com/MartinLoeper"><img src="https://github.com/MartinLoeper.png" width="60px" alt="Martin Löper" /></a>
<a href="https://github.com/lippoliv"><img src="https://github.com/lippoliv.png" width="60px" alt="Oliver Lippert" /></a>
<a href="https://github.com/lfg-seth"><img src="https://github.com/lfg-seth.png" width="60px" alt="Seth Smith" /></a>
<a href="https://github.com/snorkrat"><img src="https://github.com/snorkrat.png" width="60px" alt="" /></a>
<a href="https://github.com/spacequest-ltd"><img src="https://github.com/spacequest-ltd.png" width="60px" alt="SpaceQuest Ltd" /></a>
<a href="https://github.com/appwrite"><img src="https://github.com/appwrite.png" width="60px" alt="Appwrite" /></a>
<a href="https://github.com/PricelessToolkit"><img src="https://github.com/PricelessToolkit.png" width="60px" alt="" /></a>
<a href="https://github.com/cabottech"><img src="https://github.com/cabottech.png" width="60px" alt="Cabot Technologies" /></a>
<a href="https://github.com/markus-k"><img src="https://github.com/markus-k.png" width="60px" alt="Markus Kasten" /></a>
<a href="https://github.com/jefffhaynes"><img src="https://github.com/jefffhaynes.png" width="60px" alt="Jeff Haynes" /></a>
<a href="https://github.com/dnviti"><img src="https://github.com/dnviti.png" width="60px" alt="Daniele Viti" /></a>
<a href="https://github.com/Islendur"><img src="https://github.com/Islendur.png" width="60px" alt="Islendur" /></a>
<a href="https://github.com/Gibeon-NL"><img src="https://github.com/Gibeon-NL.png" width="60px" alt="Gibeon-NL" /></a>
<a href="https://github.com/Motrac-Research-Engineering"><img src="https://github.com/Motrac-Research-Engineering.png" width="60px" alt="Motrac Research" /></a>
<a href="https://github.com/trytuna"><img src="https://github.com/trytuna.png" width="60px" alt="Timo Scrappe" /></a>
<a href="https://github.com/ATLAS2246"><img src="https://github.com/ATLAS2246.png" width="60px" alt="ATLAS2246" /></a>
<a href="https://github.com/Kedarius"><img src="https://github.com/Kedarius.png" width="60px" alt="Radek Hladik" /></a>
</p>

<p>With ongoing resources provided by:</p>

<p align="center">
  <a href="https://depot.dev?utm_source=inventree"><img src="https://depot.dev/badges/built-with-depot.svg" alt="Built with Depot" /></a>
  <a href="https://inventree.org/digitalocean">
    <img src="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_blue.svg" width="201px" alt="Servers by Digital Ocean">
  </a>
  <a href="https://www.netlify.com"> <img src="https://www.netlify.com/v3/img/components/netlify-color-bg.svg" alt="Deploys by Netlify" /> </a>
  <a href="https://crowdin.com"> <img src="https://crowdin.com/images/crowdin-logo.svg" alt="Translation by Crowdin" /> </a> <br>
</p>

### :warning: License

InvenTree is licensed under the [MIT License](https://choosealicense.com/licenses/mit/). See [LICENSE.txt](https://github.com/inventree/InvenTree/blob/master/LICENSE) for details.