<!-- InvenTree Header Section -->
<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
  <h1>InvenTree: Open Source Inventory Management</h1>
</div>

<!-- Badges -->
<div align="center">
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

<!-- About the Project -->
## 🌟 InvenTree: The Powerful Open-Source Inventory Management System

InvenTree is a robust, open-source inventory management system designed to give you complete control over your stock control and parts tracking.  Manage your inventory effectively with its intuitive web-based interface and powerful REST API.  [Explore InvenTree on GitHub](https://github.com/inventree/InvenTree).

**Key Features:**

*   **Comprehensive Inventory Tracking:** Manage parts, stock, locations, and more with ease.
*   **Web-Based Admin Interface:** User-friendly interface for easy access and management.
*   **REST API:** Integrate with other applications and automate tasks.
*   **Customization & Extensibility:** Plugin system for custom applications and extensions.
*   **Mobile App Support:** Companion app for convenient access on the go.

<!-- Roadmap -->
### 🧭 Roadmap

Discover what's planned for InvenTree and follow the project's progress through the [roadmap tag](https://github.com/inventree/InvenTree/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap) and [horizon milestone](https://github.com/inventree/InvenTree/milestone/42).

<!-- Integration -->
### 🛠️ Integration Capabilities

InvenTree offers multiple options for seamless integration with external applications:

*   [InvenTree API](https://docs.inventree.org/en/latest/api/)
*   [Python module](https://docs.inventree.org/en/latest/api/python/)
*   [Plugin interface](https://docs.inventree.org/en/latest/plugins/)
*   [Third party tools](https://docs.inventree.org/en/latest/plugins/integrate/)

<!-- TechStack -->
### 💻 Tech Stack

<details>
  <summary>Server</summary>
  <ul>
    <li><a href="https://www.python.org/">Python</a></li>
    <li><a href="https://www.djangoproject.com/">Django</a></li>
    <li><a href="https://www.django-rest-framework.org/">DRF</a></li>
    <li><a href="https://django-q.readthedocs.io/">Django Q</a></li>
    <li><a href="https://docs.allauth.org/">Django-Allauth</a></li>
  </ul>
</details>

<details>
<summary>Database</summary>
  <ul>
    <li><a href="https://www.postgresql.org/">PostgreSQL</a></li>
    <li><a href="https://www.mysql.com/">MySQL</a></li>
    <li><a href="https://www.sqlite.org/">SQLite</a></li>
    <li><a href="https://redis.io/">Redis</a></li>
  </ul>
</details>

<details>
  <summary>Client</summary>
  <ul>
    <li><a href="https://react.dev/">React</a></li>
    <li><a href="https://lingui.dev/">Lingui</a></li>
    <li><a href="https://reactrouter.com/">React Router</a></li>
    <li><a href="https://tanstack.com/query/">TanStack Query</a></li>
    <li><a href="https://github.com/pmndrs/zustand">Zustand</a></li>
    <li><a href="https://mantine.dev/">Mantine</a></li>
    <li><a href="https://icflorescu.github.io/mantine-datatable/">Mantine Data Table</a></li>
    <li><a href="https://codemirror.net/">CodeMirror</a></li>
  </ul>
</details>

<details>
<summary>DevOps</summary>
  <ul>
    <li><a href="https://hub.docker.com/r/inventree/inventree">Docker</a></li>
    <li><a href="https://crowdin.com/project/inventree">Crowdin</a></li>
    <li><a href="https://app.codecov.io/gh/inventree/InvenTree">Codecov</a></li>
    <li><a href="https://sonarcloud.io/project/overview?id=inventree_InvenTree">SonarCloud</a></li>
    <li><a href="https://packager.io/gh/inventree/InvenTree">Packager.io</a></li>
  </ul>
</details>

<!-- Getting Started -->
## 🚀 Getting Started

InvenTree offers several deployment options to suit your needs.

<div align="center"><h4>
    <a href="https://docs.inventree.org/en/latest/start/docker/">Docker</a>
    <span> · </span>
    <a href="https://inventree.org/digitalocean"><img src="https://www.deploytodo.com/do-btn-blue-ghost.svg" alt="Deploy to DO" width="auto" height="40" /></a>
    <span> · </span>
    <a href="https://docs.inventree.org/en/latest/start/install/">Bare Metal</a>
</h4></div>

**Quick Install:**

```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```
Refer to the [getting started guide](https://docs.inventree.org/en/latest/start/install/) for complete instructions.

<!-- Mobile App -->
## 📱 Mobile App

Access and manage your inventory on the go with the [InvenTree companion mobile app](https://docs.inventree.org/app/).

<div align="center"><h4>
    <a href="https://play.google.com/store/apps/details?id=inventree.inventree_app">Android Play Store</a>
     <span> · </span>
    <a href="https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone">Apple App Store</a>
</h4></div>

<!-- Security -->
## 🔒 Security & Code of Conduct

The InvenTree project is committed to a safe and welcoming environment. Review our [Code of Conduct](CODE_OF_CONDUCT.md) and security policy [in this repo](SECURITY.md). Find dedicated security information on our [documentation site](https://docs.inventree.org/en/latest/security/).

<!-- Contributing -->
## 🤝 Contributing

Join the InvenTree community and contribute to its development!  Visit the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/) for details.

<!-- Translation -->
## 🌐 Translation

Help translate the InvenTree web application into your native language via [Crowdin](https://crowdin.com/project/inventree).

<!-- Sponsor -->
## 💖 Sponsor

Support the project's development by [sponsoring InvenTree](https://github.com/sponsors/inventree).

<!-- Acknowledgments -->
## ✨ Acknowledgements

We thank [PartKeepr](https://github.com/partkeepr/PartKeepr) for its inspiration.  See the license information dialog for a full list of third-party libraries used.

<!-- Support -->
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

<!-- License -->
## 📜 License

InvenTree is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).
```

Key improvements and SEO optimizations:

*   **Clear, Concise Title:**  Uses "InvenTree: Open Source Inventory Management" for better keyword targeting.
*   **One-Sentence Hook:**  Highlights the main benefit to users.
*   **Keywords:**  "Inventory Management," "Open Source," and related terms are used naturally throughout.
*   **Headings:**  Uses H2 and H3 headings for better structure and readability.
*   **Bulleted Key Features:** Makes it easy for users to quickly grasp the value proposition.
*   **Links to key areas:** Provides direct links to important pages within the documentation and repository.
*   **Clear Call to Action:** Encourages users to explore the demo and contribute.
*   **Concise Descriptions:** Avoids overly verbose explanations.
*   **SEO-Friendly Formatting:**  Uses bolding and other formatting to emphasize important information.
*   **Includes the original links**
*   **Sponsor and Support sections were updated and kept**