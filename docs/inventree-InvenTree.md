<!-- InvenTree Logo -->
<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
</div>

<!-- Title and Description -->
<h1>InvenTree: Open Source Inventory Management System</h1>
<p><b>InvenTree is your all-in-one solution for efficient inventory management, stock control, and part tracking.</b></p>

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

<!-- Quick Links -->
<div align="center">
  <a href="https://demo.inventree.org/">View Demo</a>
  <span> · </span>
  <a href="https://docs.inventree.org/en/latest/">Documentation</a>
  <span> · </span>
  <a href="https://github.com/inventree/InvenTree/issues/new?template=bug_report.md&title=[BUG]">Report Bug</a>
  <span> · </span>
  <a href="https://github.com/inventree/InvenTree/issues/new?template=feature_request.md&title=[FR]">Request Feature</a>
</div>

<!-- About the Project -->
## 🎯 About InvenTree

InvenTree is a powerful, open-source inventory management system designed for streamlined stock control and precise part tracking. Built on a robust Python/Django backend, InvenTree offers a user-friendly web-based admin interface and a comprehensive REST API, facilitating seamless integration with external systems.  Enhance your workflow with a flexible plugin system, allowing for custom applications and tailored extensions.

*   **Open Source:** Fully accessible and customizable under the MIT License.
*   **Web-Based Interface:** Intuitive and easy-to-use for all your inventory needs.
*   **REST API:** Integrate with other systems and automate your workflows.
*   **Plugin System:** Extend functionality with custom applications and extensions.
*   **Part Tracking:** Maintain detailed records of all your parts and components.

Explore more on the [InvenTree website](https://inventree.org) and find the source code on [GitHub](https://github.com/inventree/InvenTree).

<!-- Roadmap -->
### 🚀 Roadmap & Future Plans

Stay informed about the future of InvenTree.  View our [roadmap tag](https://github.com/inventree/InvenTree/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap) and [horizon milestone](https://github.com/inventree/InvenTree/milestone/42) to see what features are being developed and planned.

<!-- Integration -->
### 🔌 Integration & Extensibility

InvenTree's design emphasizes extensibility, providing various integration options for external applications and the creation of custom plugins:

*   [InvenTree API](https://docs.inventree.org/en/latest/api/)
*   [Python module](https://docs.inventree.org/en/latest/api/python/)
*   [Plugin interface](https://docs.inventree.org/en/latest/plugins/)
*   [Third party tools](https://docs.inventree.org/en/latest/plugins/integrate/)

<!-- TechStack -->
### 🛠️ Tech Stack

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
## ⚙️ Getting Started & Deployment

Choose the deployment method that best suits your needs:

<div align="center">
  <a href="https://docs.inventree.org/en/latest/start/docker/">Docker</a>
  <span> · </span>
  <a href="https://inventree.org/digitalocean"><img src="https://www.deploytodo.com/do-btn-blue-ghost.svg" alt="Deploy to DO" width="auto" height="40" /></a>
  <span> · </span>
  <a href="https://docs.inventree.org/en/latest/start/install/">Bare Metal</a>
</div>

For a quick setup, use the one-line installer:
```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

For complete installation instructions, refer to the [getting started guide](https://docs.inventree.org/en/latest/start/install/).

<!-- Mobile App -->
## 📱 Mobile App

Access InvenTree on the go with our companion mobile app, available for both Android and iOS:

<div align="center">
  <a href="https://play.google.com/store/apps/details?id=inventree.inventree_app">Android Play Store</a>
  <span> · </span>
  <a href="https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone">Apple App Store</a>
</div>

<!-- Security -->
## 🔒 Security & Code of Conduct

We are committed to security and a welcoming community. Review our [Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md) for more details.  You can also find dedicated security pages on [our documentation site](https://docs.inventree.org/en/latest/security/).

<!-- Contributing -->
## 🤝 Contributing

We welcome contributions from the community!  Learn how to contribute by visiting the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/).

<!-- Translation -->
## 🌐 Translation

Help us make InvenTree accessible worldwide by contributing to the native language translations via [Crowdin](https://crowdin.com/project/inventree). **Contributions are greatly appreciated!**

<!-- Sponsor -->
## 💖 Sponsor

Support the development of InvenTree by becoming a [sponsor](https://github.com/sponsors/inventree).

<!-- Acknowledgments -->
## ✨ Acknowledgements

We acknowledge [PartKeepr](https://github.com/partkeepr/PartKeepr) as a source of inspiration.  See the license information dialog of your instance for a complete list of third-party libraries.

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
## 📝 License

InvenTree is distributed under the [MIT](https://choosealicense.com/licenses/mit/) License.  See [LICENSE.txt](https://github.com/inventree/InvenTree/blob/master/LICENSE) for details.

<!-- Back to Top Link -->
<p align="right"><a href="#top">Back to Top</a></p>
```

Key improvements and SEO considerations:

*   **Clear Hook:**  The first sentence clearly states the value proposition (inventory management).
*   **Keyword Optimization:**  Keywords like "Inventory Management System," "Stock Control," and "Part Tracking" are used naturally throughout the document.
*   **Headings:**  Uses H2 and H3 headings to structure the document and improve readability for both users and search engines.
*   **Bulleted Lists:**  Key features are presented in bulleted lists for easy scanning.
*   **Clear Calls to Action:** Encourages users to visit the demo, documentation, and report bugs.
*   **Concise Language:**  Streamlines the text for better comprehension.
*   **Relevant Links:** Includes links to the GitHub repository, documentation, and other important resources.
*   **Mobile App Section:** Highlights the mobile app functionality.
*   **SEO-Friendly Formatting:**  Uses Markdown headings and formatting for SEO.
*   **Back to Top link:** Adds a back to top link for easier navigation.