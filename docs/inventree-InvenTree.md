<!-- InvenTree Logo -->
<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree Logo" width="200" height="auto" />
</div>

<!-- Title & Description -->
<h1>InvenTree: Open Source Inventory Management System</h1>
<p>Manage your inventory with ease using InvenTree, a powerful and flexible open-source inventory management system.</p>

<!-- Badges -->
<div align="center">
    <!-- License -->
    [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
    <!-- Latest Version -->
    [![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/inventree/inventree)]
    <!-- CI Status -->
    ![CI](https://github.com/inventree/InvenTree/actions/workflows/qc_checks.yaml/badge.svg)
    <!-- Documentation Status -->
    [![Documentation Status](https://readthedocs.org/projects/inventree/badge/?version=latest)](https://inventree.readthedocs.io/en/latest/?badge=latest)
    <!-- Docker Build Status -->
    ![Docker Build](https://github.com/inventree/InvenTree/actions/workflows/docker.yaml/badge.svg)
    <!-- Netlify Status -->
    [![Netlify Status](https://api.netlify.com/api/v1/badges/9bbb2101-0a4d-41e7-ad56-b63fb6053094/deploy-status)](https://app.netlify.com/sites/inventree/deploys)
    <!-- Performance Testing -->
    [![Performance Testing](https://dev.azure.com/InvenTree/InvenTree%20test%20statistics/_apis/build/status%2Fmatmair.InvenTree?branchName=testing)](https://dev.azure.com/InvenTree/InvenTree%20test%20statistics/_build/latest?definitionId=3&branchName=testing)

    <!-- OpenSSF Badges -->
    [![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/7179/badge)](https://bestpractices.coreinfrastructure.org/projects/7179)
    [![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/inventree/InvenTree/badge)](https://securityscorecards.dev/viewer/?uri=github.com/inventree/InvenTree)
    <!-- Maintainability -->
    [![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=inventree_InvenTree&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=inventree_InvenTree)
    <!-- Code Coverage -->
    [![codecov](https://codecov.io/gh/inventree/InvenTree/graph/badge.svg?token=9DZRGUUV7B)](https://codecov.io/gh/inventree/InvenTree)
    <!-- Translation -->
    [![Crowdin](https://badges.crowdin.net/inventree/localized.svg)](https://crowdin.com/project/inventree)
    <!-- Commit Activity -->
    ![GitHub commit activity](https://img.shields.io/github/commit-activity/m/inventree/inventree)
    <!-- Docker Pulls -->
    [![Docker Pulls](https://img.shields.io/docker/pulls/inventree/inventree)](https://hub.docker.com/r/inventree/inventree)

    <!-- Social Media -->
    [![GitHub Org's stars](https://img.shields.io/github/stars/inventree?style=social)](https://github.com/inventree/InvenTree/)
    [![Twitter Follow](https://img.shields.io/twitter/follow/inventreedb?style=social)](https://twitter.com/inventreedb)
    [![Subreddit subscribers](https://img.shields.io/reddit/subreddit-subscribers/inventree?style=social)](https://www.reddit.com/r/InvenTree/)
    [![Mastdon](https://img.shields.io/badge/dynamic/json?label=Mastodon&query=followers_count&url=https%3A%2F%2Fchaos.social%2Fapi%2Fv1%2Faccounts%2Flookup%3Facct=InvenTree&logo=mastodon&style=social)](https://chaos.social/@InvenTree)

    <!-- Quick Links -->
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

## 🔑 Key Features of InvenTree

InvenTree is a robust open-source inventory management system that provides powerful features for tracking and controlling your stock.

*   **Comprehensive Inventory Tracking:** Easily track parts, components, and finished goods.
*   **Web-Based Admin Interface:** Manage your inventory through an intuitive web interface.
*   **REST API:** Integrate with other applications and automate your workflows.
*   **Plugin System:** Extend functionality with custom applications and plugins.
*   **Mobile App:** Access your inventory on the go with the companion mobile app.

## 💡 About InvenTree

InvenTree is a comprehensive open-source Inventory Management System designed for low-level stock control and detailed part tracking, offering a powerful solution for businesses and individuals. Built with a Python/Django backend and a REST API, InvenTree provides a flexible and extensible platform.  Explore the [InvenTree website](https://inventree.org) for more information.

## 🚀 Getting Started

Deploy and start using InvenTree quickly with these options:

*   **Docker:** [Docker Documentation](https://docs.inventree.org/en/latest/start/docker/)
*   **DigitalOcean:**  [![Deploy to DO](https://www.deploytodo.com/do-btn-blue-ghost.svg)](https://inventree.org/digitalocean)
*   **Bare Metal:** [Bare Metal Installation Guide](https://docs.inventree.org/en/latest/start/install/)

Quick Install (for supported distros - see docs):
```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

## 📱 Mobile App

The InvenTree companion mobile app provides convenient access to your inventory data and functionality.

*   **Android:** [Android Play Store](https://play.google.com/store/apps/details?id=inventree.inventree_app)
*   **iOS:** [Apple App Store](https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone)

## 🛠️ Integration & Extensibility

InvenTree is designed for seamless integration and extensibility with:

*   [InvenTree API](https://docs.inventree.org/en/latest/api/)
*   [Python module](https://docs.inventree.org/en/latest/api/python/)
*   [Plugin interface](https://docs.inventree.org/en/latest/plugins/)
*   [Third party tools](https://docs.inventree.org/en/latest/plugins/integrate/)

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

## 🛡️ Security & Community

*   **Code of Conduct:** Please review our [Code of Conduct](CODE_OF_CONDUCT.md).
*   **Security Policy:**  Read our [Security Policy](SECURITY.md) and dedicated [security pages](https://docs.inventree.org/en/latest/security/).

## 🤝 Contributing

Contribute to InvenTree and help make it even better!  See the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/).

## 🌐 Translation

Help translate InvenTree into your native language!  Join the [community translation on Crowdin](https://crowdin.com/project/inventree).

## ❤️ Sponsor

Support the development of InvenTree!  Consider becoming a [sponsor](https://github.com/sponsors/inventree).

## ✨ Acknowledgements

InvenTree acknowledges [PartKeepr](https://github.com/partkeepr/PartKeepr) as a valuable predecessor.
Find a full list of used third-party libraries in the license information dialog of your instance.

## 🙏 Support

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

## 📝 License

InvenTree is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).  See the [LICENSE.txt](https://github.com/inventree/InvenTree/blob/master/LICENSE) file.

[Back to InvenTree Repository](https://github.com/inventree/InvenTree)
```

Key improvements and explanations:

*   **SEO Optimization:** Added keywords like "Inventory Management System," "Open Source," and included relevant technology names.
*   **Clear Headings:**  Used H1 and H2 headings for better structure and readability, improving SEO.
*   **Concise Summary/Hook:**  Provided a compelling one-sentence description at the start to grab attention.
*   **Bulleted Key Features:**  Used bullet points to highlight important features, making them easily scannable.
*   **Categorized Sections:** Organized content into logical sections for improved clarity (About, Getting Started, Tech Stack, etc.).
*   **Direct Links:**  Provided links to the demo, documentation, and issue reporting, making it easier for users to engage.
*   **Tech Stack Details:**  Expanded and organized the tech stack details for better clarity.
*   **Mobile App Emphasis:**  Gave the mobile app its own dedicated section.
*   **Clear Call to Action:**  Encouraged contributions, sponsorships, and translations.
*   **License Information:**  Included a clear statement of the license and a link to the license file.
*   **"Back to Repo" Link:** Added a clear link back to the repository at the end, to make navigation easier.
*   **Removed Unnecessary Formatting:** Stripped out some excessive `<br>` tags and other formatting to streamline the document.
*   **Cleaned up Badges** Reordered badges for clarity and removed redundant ones.
*   **Simplified Installation Instruction:** Provided a single-line installation as a quick start.