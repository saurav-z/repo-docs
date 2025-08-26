<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
  <h1>InvenTree</h1>
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
</a>

<!-- About the Project -->
## 🚀 InvenTree: The Open-Source Inventory Management System

InvenTree is a powerful, open-source inventory management system designed to streamline stock control and part tracking.

**Key Features:**

*   **Web-Based Admin Interface:** Manage your inventory through an intuitive web interface.
*   **REST API:** Integrate with external applications using a robust REST API.
*   **Part Tracking:** Efficiently track individual parts and components.
*   **Plugin System:** Extend functionality with custom plugins.
*   **Mobile App:** Access and manage inventory on the go with the companion mobile app.
*   **Open Source:** Benefit from community contributions and full control over your data.
*   **Extensible:** Integrate with other applications or add custom plugins.
*   **Multi-Platform Support:** Deploy using Docker, bare metal, or cloud solutions.

Learn more about InvenTree on the [official website](https://inventree.org).

<!-- Getting Started -->
## ⚙️ Getting Started

Choose your preferred deployment method:

*   **Docker:** [Docker Documentation](https://docs.inventree.org/en/latest/start/docker/)
*   **DigitalOcean:** <a href="https://inventree.org/digitalocean"><img src="https://www.deploytodo.com/do-btn-blue-ghost.svg" alt="Deploy to DO" width="auto" height="40" /></a>
*   **Bare Metal:** [Installation Guide](https://docs.inventree.org/en/latest/start/install/)

Install with a single line using the following:
```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

For detailed installation and setup instructions, refer to the [getting started guide](https://docs.inventree.org/en/latest/start/install/).

<!-- Mobile App -->
## 📱 Mobile App

InvenTree offers a companion mobile app for easy inventory access and management:

*   [Android Play Store](https://play.google.com/store/apps/details?id=inventree.inventree_app)
*   [Apple App Store](https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone)

<!-- Integration -->
## 🔌 Integration and Extensibility

InvenTree is designed for seamless integration and extensibility:

*   [InvenTree API](https://docs.inventree.org/en/latest/api/)
*   [Python module](https://docs.inventree.org/en/latest/api/python/)
*   [Plugin interface](https://docs.inventree.org/en/latest/plugins/)
*   [Third party tools](https://docs.inventree.org/en/latest/plugins/integrate/)

<!-- Technology Stack -->
## 💻 Tech Stack

**Server:**

*   Python
*   Django
*   Django REST Framework (DRF)
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

<!-- Security -->
## 🔒 Security and Community

*   **Code of Conduct:** Read our [Code of Conduct](CODE_OF_CONDUCT.md) to learn more about our community guidelines.
*   **Security Policy:** Review our [security policy](SECURITY.md) for best practices.
*   Dedicated security pages are available on [our documentation site](https://docs.inventree.org/en/latest/security/).

<!-- Contributing -->
## 🤝 Contributing

Contribute to the InvenTree project and help us improve it! Visit the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/) to get started.

<!-- Translation -->
## 🌐 Translation

Help translate the InvenTree web application into your native language via [Crowdin](https://crowdin.com/project/inventree).

<!-- Sponsor -->
## 💖 Sponsorship

Support the InvenTree project by becoming a sponsor: [Sponsor InvenTree](https://github.com/sponsors/inventree).

<!-- Acknowledgments -->
## ✨ Acknowledgments

We are grateful to [PartKeepr](https://github.com/partkeepr/PartKeepr) for their inspiration. Find a full list of third-party libraries in the license information dialog.

## :heart: Support

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

InvenTree is licensed under the [MIT License](https://choosealicense.com/licenses/mit/). Access the full license [here](https://github.com/inventree/InvenTree/blob/master/LICENSE).

<!-- Back to Top -->
<p align="right"><a href="#top">Back to Top</a></p>

```

Key improvements and SEO optimizations:

*   **Concise Hook:** Starts with a strong, benefit-driven hook.
*   **Keyword Optimization:** Uses relevant keywords throughout (Inventory Management, Open Source, Stock Control, Part Tracking).
*   **Clear Headings:** Uses headings and subheadings to improve readability and structure for both users and search engines.
*   **Bulleted Key Features:** Highlights key features for quick scanning.
*   **Clear Call to Action:** Includes links to demo, documentation, bug reporting and feature request.
*   **Link Back to Repo:** Ensures the original repo is easily accessible.
*   **Improved Readability:** Uses better formatting and spacing.
*   **Includes relevant badges**: To show CI/CD, security, etc.
*   **Added Back to Top:** for better user experience.

This revised README is much more informative, user-friendly, and search engine optimized than the original.