<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
  <h1>InvenTree</h1>
  <p>Open Source Inventory Management System </p>
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

<!-- About the Project -->
## 📦 InvenTree: The Open-Source Inventory Management Solution

**InvenTree is your powerful, open-source solution for streamlining inventory tracking and part management.** ([View the GitHub Repository](https://github.com/inventree/InvenTree))

*   **Web-Based Interface:** Access and manage your inventory from anywhere with a user-friendly web interface.
*   **Part Tracking:** Detailed part tracking and stock control functionality.
*   **REST API:** Integrate with other applications using the REST API.
*   **Plugin System:** Extend functionality with custom plugins.
*   **Mobile App Support:** Companion mobile app for on-the-go inventory management.
*   **Extensive Documentation:** Comprehensive documentation to guide you.
*   **Community Support:** Benefit from a vibrant and supportive community.

<!-- Getting Started -->
## 🚀 Getting Started

Deploy InvenTree with ease using Docker, DigitalOcean, or bare metal installations.  Visit the documentation for a full set of installation and setup instructions.

*   **Docker:** <a href="https://docs.inventree.org/en/latest/start/docker/">Docker Instructions</a>
*   **DigitalOcean:** <a href="https://inventree.org/digitalocean"><img src="https://www.deploytodo.com/do-btn-blue-ghost.svg" alt="Deploy to DO" width="auto" height="40" /></a>
*   **Bare Metal:** <a href="https://docs.inventree.org/en/latest/start/install/">Bare Metal Installation</a>

Quick installation using a single command:

```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

<!-- Key Features -->
## ✨ Key Features

*   **Part and Stock Management:** Comprehensive tracking of parts, stock levels, and locations.
*   **BOM Management:** Build-of-materials management for tracking components used in assemblies.
*   **Purchase Order & Sales Order:** Manage purchase orders, sales orders, and associated transactions.
*   **Barcode Scanning:** Built-in support for barcode scanning.
*   **User Roles and Permissions:** Granular control over user access and permissions.
*   **Reporting and Analytics:** Generate reports to track inventory trends.
*   **Customizable Fields:** Add custom fields to parts, stock items, and other data.
*   **API and Plugin Support:** Integrate with other systems and extend functionality.

<!-- Mobile App -->
## 📱 Mobile App

Manage your inventory on the go with the InvenTree companion mobile app.

*   **Android:** <a href="https://play.google.com/store/apps/details?id=inventree.inventree_app">Android Play Store</a>
*   **iOS:** <a href="https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone">Apple App Store</a>

<!-- Integration -->
## 🔌 Integration

InvenTree is designed to be extensible and provides multiple options for integration.

*   [InvenTree API](https://docs.inventree.org/en/latest/api/)
*   [Python module](https://docs.inventree.org/en/latest/api/python/)
*   [Plugin interface](https://docs.inventree.org/en/latest/plugins/)
*   [Third party tools](https://docs.inventree.org/en/latest/plugins/integrate/)

<!-- TechStack -->
## 💻 Tech Stack

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

<!-- Code of Conduct & Security -->
## 🛡️ Code of Conduct & Security

InvenTree is committed to a safe and welcoming environment.

*   **Code of Conduct:** [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
*   **Security Policy:** [SECURITY.md](SECURITY.md)
*   **Security Documentation:** [Security Documentation](https://docs.inventree.org/en/latest/security/)

<!-- Contributing -->
## 🙌 Contributing

Contributions are welcome!  Help make InvenTree even better.

*   **Contribution Guide:** [Contribution Page](https://docs.inventree.org/en/latest/develop/contributing/)

<!-- Translation -->
## 🌐 Translation

Help translate InvenTree into your native language through community contributions via [Crowdin](https://crowdin.com/project/inventree).

<!-- Sponsor -->
## 🙏 Sponsor

Support the InvenTree project by becoming a sponsor! [Sponsor InvenTree](https://github.com/sponsors/inventree)

<!-- Acknowledgments -->
## 💖 Acknowledgments

InvenTree is inspired by [PartKeepr](https://github.com/partkeepr/PartKeepr).

Find a full list of third-party libraries in the license information dialog of your instance.

## 💬 Support

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

<!-- License -->
## 📜 License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See [LICENSE.txt](https://github.com/inventree/InvenTree/blob/master/LICENSE) for more information.