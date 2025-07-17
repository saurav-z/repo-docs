<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
  <h1>InvenTree</h1>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/inventree/inventree)]()
[![CI](https://github.com/inventree/InvenTree/actions/workflows/qc_checks.yaml/badge.svg)]()
[![Documentation Status](https://readthedocs.org/projects/inventree/badge/?version=latest)]()
[![Docker Build](https://github.com/inventree/InvenTree/actions/workflows/docker.yaml/badge.svg)]()
[![Netlify Status](https://api.netlify.com/api/v1/badges/9bbb2101-0a4d-41e7-ad56-b63fb6053094/deploy-status)]()
[![Performance Testing](https://dev.azure.com/InvenTree/InvenTree%20test%20statistics/_apis/build/status%2Fmatmair.InvenTree?branchName=testing)]()

[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/7179/badge)]()
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/inventree/InvenTree/badge)]()
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=inventree_InvenTree&metric=sqale_rating)]()

[![codecov](https://codecov.io/gh/inventree/InvenTree/graph/badge.svg?token=9DZRGUUV7B)]()
[![Crowdin](https://badges.crowdin.net/inventree/localized.svg)]()
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/inventree/inventree)]()
[![Docker Pulls](https://img.shields.io/docker/pulls/inventree/inventree)]()

[![GitHub Org's stars](https://img.shields.io/github/stars/inventree?style=social)]()
[![Twitter Follow](https://img.shields.io/twitter/follow/inventreedb?style=social)]()
[![Subreddit subscribers](https://img.shields.io/reddit/subreddit-subscribers/inventree?style=social)]()
[![Mastdon](https://img.shields.io/badge/dynamic/json?label=Mastodon&query=followers_count&url=https%3A%2F%2Fchaos.social%2Fapi%2Fv1%2Faccounts%2Flookup%3Facct=InvenTree&logo=mastodon&style=social)]()

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
## 📦 InvenTree: Open Source Inventory Management for Streamlined Stock Control

InvenTree is a powerful, open-source inventory management system designed to help you efficiently track parts, manage stock levels, and optimize your supply chain.  Visit the [InvenTree GitHub repository](https://github.com/inventree/InvenTree) to get started.

**Key Features:**

*   **Comprehensive Stock Control:** Manage parts, track stock levels, and monitor inventory movements.
*   **Web-Based Admin Interface:** User-friendly interface for easy access and management.
*   **REST API:** Integrate with other systems and applications using a robust API.
*   **Plugin System:** Extend functionality with custom applications and extensions.
*   **Mobile App:** Companion app for Android and iOS devices for on-the-go access.
*   **Extensive Documentation:** Detailed documentation to guide you through setup, use, and customization.
*   **Multiple Deployment Options:** Deploy via Docker, DigitalOcean, or bare metal installations.
*   **Multilingual Support:**  Community-driven translation for global accessibility.

<!-- Roadmap -->
### 🧭 Roadmap

*   Explore the [roadmap](https://github.com/inventree/InvenTree/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap) to see what's coming and the [horizon milestone](https://github.com/inventree/InvenTree/milestone/42).

<!-- Integration -->
### 🛠️ Integration

InvenTree is designed to be **extensible**, and provides multiple options for **integration** with external applications or addition of custom plugins:

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

InvenTree offers multiple deployment methods for easy setup.

<div align="center"><h4>
    <a href="https://docs.inventree.org/en/latest/start/docker/">Docker</a>
    <span> · </span>
    <a href="https://inventree.org/digitalocean"><img src="https://www.deploytodo.com/do-btn-blue-ghost.svg" alt="Deploy to DO" width="auto" height="40" /></a>
    <span> · </span>
    <a href="https://docs.inventree.org/en/latest/start/install/">Bare Metal</a>
</h4></div>

Quick install via command line:
```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

For detailed setup, follow the [getting started guide](https://docs.inventree.org/en/latest/start/install/).

<!-- Mobile App -->
## 📱 Mobile App

Access your inventory on the go with the InvenTree companion mobile app:

<div align="center"><h4>
    <a href="https://play.google.com/store/apps/details?id=inventree.inventree_app">Android Play Store</a>
     <span> · </span>
    <a href="https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone">Apple App Store</a>
</h4></div>

<!-- Security -->
## 🔒 Security & Community

*   Review our [Code of Conduct](CODE_OF_CONDUCT.md).
*   Read our [Security Policy](SECURITY.md) and [security documentation](https://docs.inventree.org/en/latest/security/).

<!-- Contributing -->
## 🤝 Contributing

Contribute to the project!  See the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/) for details.

<!-- Translation -->
## 🌐 Translation

Help localize InvenTree!  Translate the web application [via Crowdin](https://crowdin.com/project/inventree).

<!-- Sponsor -->
## 💖 Sponsor

Support InvenTree's development and maintenance by [sponsoring the project](https://github.com/sponsors/inventree).

<!-- Acknowledgments -->
## 🙏 Acknowledgements

Special thanks to [PartKeepr](https://github.com/partkeepr/PartKeepr) for inspiration.

Find a full list of third-party libraries in your instance's license information dialog.

## ❤️ Support

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
```
Key improvements and SEO considerations:

*   **Clear Headings:** Uses descriptive headings (H2, H3) to organize content logically and improve readability.
*   **Keyword Optimization:**  Includes relevant keywords like "Inventory Management," "Stock Control," "Open Source," and mentions the key features.
*   **Concise and Engaging Hook:** The one-sentence summary quickly grabs attention and highlights the core value proposition.
*   **Bulleted Key Features:** Uses bullet points to make the main benefits easy to scan.
*   **Internal Linking:**  Links to key sections like the Roadmap, Integration, and Getting Started.
*   **External Linking:** Includes relevant links to the website, documentation, and other resources.
*   **Call to Action:**  Encourages users to "get started" and contribute.
*   **SEO-Friendly Formatting:** Uses bold text strategically.
*   **Concise Language:**  Avoids jargon and uses clear, straightforward language.
*   **License Section:** Explicitly mentions the license.
*   **Consistent formatting:**  Corrected alignment and spacing for readability.