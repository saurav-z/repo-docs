<!-- InvenTree Banner -->
<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
  <h1>InvenTree: Open Source Inventory Management System</h1>
  <p>Effectively manage your inventory with this powerful and open-source solution. <a href="https://github.com/inventree/InvenTree">View the code on GitHub</a>.</p>

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

<!-- About the Project -->
## :star2: About InvenTree: Your Open Source Inventory Solution

InvenTree is a powerful and versatile open-source inventory management system designed to streamline stock control and part tracking. Whether you're managing a small workshop or a large enterprise, InvenTree provides the tools you need to efficiently manage your parts and inventory.

**Key Features:**

*   **Web-Based Admin Interface:** Manage your inventory through an intuitive web-based admin interface.
*   **REST API:** Integrate with external applications and automate tasks using the REST API.
*   **Plugin System:** Extend InvenTree's functionality with a robust plugin system for custom applications and integrations.
*   **Part Tracking:** Keep detailed records of your parts, including specifications, suppliers, and stock levels.
*   **Stock Control:** Monitor and manage your stock levels to prevent shortages and overstocking.

**Learn more:** [our website](https://inventree.org)

<!-- Roadmap -->
### :compass: Roadmap

Explore InvenTree's future! Check out the [roadmap tag](https://github.com/inventree/InvenTree/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap) and [horizon milestone](https://github.com/inventree/InvenTree/milestone/42).

<!-- Integration -->
### :hammer_and_wrench: Integration & Extensibility

InvenTree is designed for seamless integration and offers multiple options for adding custom plugins:

*   [InvenTree API](https://docs.inventree.org/en/latest/api/)
*   [Python module](https://docs.inventree.org/en/latest/api/python/)
*   [Plugin interface](https://docs.inventree.org/en/latest/plugins/)
*   [Third party tools](https://docs.inventree.org/en/latest/plugins/integrate/)

<!-- TechStack -->
### :space_invader: Technology Stack

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

<!-- Getting Started -->
## 	:toolbox: Getting Started: Deployment Options

Get started with InvenTree using one of these deployment options:

<div align="center">
<h4>
    <a href="https://docs.inventree.org/en/latest/start/docker/">Docker</a>
    <span> · </span>
    <a href="https://inventree.org/digitalocean"><img src="https://www.deploytodo.com/do-btn-blue-ghost.svg" alt="Deploy to DO" width="auto" height="40" /></a>
    <span> · </span>
    <a href="https://docs.inventree.org/en/latest/start/install/">Bare Metal</a>
</h4>
</div>

**Quick Install:**

```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

For comprehensive installation and setup instructions, refer to the [getting started guide](https://docs.inventree.org/en/latest/start/install/).

<!-- Mobile App -->
## 	:iphone: Mobile App

Enhance your InvenTree experience with our companion mobile app, giving you access to stock control information and functionality on the go.

<div align="center">
<h4>
    <a href="https://play.google.com/store/apps/details?id=inventree.inventree_app">Android Play Store</a>
     <span> · </span>
    <a href="https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone">Apple App Store</a>
</h4>
</div>

<!-- Security -->
## :lock: Security and Community Guidelines

The InvenTree project prioritizes security and community well-being.

*   Read our [Code of Conduct](CODE_OF_CONDUCT.md).
*   Review our [Security Policy](SECURITY.md).
*   Access dedicated security information on our [documentation site](https://docs.inventree.org/en/latest/security/).

<!-- Contributing -->
## :wave: Contribute to InvenTree

We welcome and encourage contributions!  Refer to the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/) for details on how to get involved.

<!-- Translation -->
## :scroll: Translation

Help translate InvenTree into your native language via [Crowdin](https://crowdin.com/project/inventree).

<!-- Sponsor -->
## :money_with_wings: Sponsor InvenTree

Support the project by [sponsoring us](https://github.com/sponsors/inventree).

<!-- Acknowledgments -->
## :gem: Acknowledgements

We want to acknowledge [PartKeepr](https://github.com/partkeepr/PartKeepr) as a valuable predecessor and inspiration.
Find a full list of used third-party libraries in the license information dialog of your instance.

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
## :warning: License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See [LICENSE.txt](https://github.com/inventree/InvenTree/blob/master/LICENSE) for more information.
```

Key improvements and SEO considerations:

*   **Clear Hook:** The first sentence immediately explains what InvenTree *is* and its primary benefit (inventory management).
*   **Keyword Optimization:**  Includes relevant keywords throughout the text like "inventory management," "open source," "stock control," "part tracking," and "inventory system."
*   **Structured Headings:**  Uses clear and descriptive headings (H2, H3) to improve readability and SEO.
*   **Bulleted Key Features:** Highlights the most important features using bullet points, making it easy for users to scan and understand the value proposition.
*   **Concise Language:** Streamlined the text for better readability.
*   **Calls to Action:** Encourages users to view the code on GitHub, check out the demo, and explore the documentation.
*   **Internal Linking:** Uses internal links to the roadmap and the documentation to improve navigation and SEO.
*   **External Links:** Added an explicit link to the original repo to attribute.
*   **Alt Text:** Improved all the image `alt` text.
*   **Emphasis on benefits.** Changed the description to focus on the problem InvenTree solves.