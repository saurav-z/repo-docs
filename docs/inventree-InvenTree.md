<!-- InvenTree Logo -->
<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree Logo" width="200" height="auto" />
</div>

<!-- Title and Description -->
<h1 align="center">InvenTree: Open Source Inventory Management System</h1>

<p align="center">
  **InvenTree is your all-in-one solution for managing parts, stock, and tracking inventory, empowering you to take control of your supply chain.**
  <br>
  <a href="https://github.com/inventree/InvenTree">View on GitHub</a>
</p>

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
</div>

<!-- Links -->
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
## :star2: About InvenTree

InvenTree is a robust, open-source inventory management system designed for efficient parts tracking, stock control, and overall supply chain management. Built with a Python/Django backend, it offers a user-friendly web-based admin interface and a powerful REST API for seamless integration with other systems. The flexible plugin system allows for extensive customization and extensions to fit your specific needs.

**Key Features:**

*   **Comprehensive Inventory Management:** Track parts, components, and stock levels with precision.
*   **Web-Based Interface:** Accessible and easy-to-use interface for managing your inventory from anywhere.
*   **REST API:** Integrate with external applications and systems for data exchange and automation.
*   **Plugin System:** Extend functionality with custom plugins and integrations.
*   **Mobile App Support:** Companion mobile app for convenient access on the go.
*   **Extensive Documentation:** Detailed documentation to guide you through setup, usage, and development.

### :compass: Roadmap

Stay up-to-date with the latest developments by checking out the [roadmap tag](https://github.com/inventree/InvenTree/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap) and [horizon milestone](https://github.com/inventree/InvenTree/milestone/42).

<!-- Integration -->
### :hammer_and_wrench: Integration

InvenTree offers several options for seamless integration with other applications:

*   [InvenTree API](https://docs.inventree.org/en/latest/api/)
*   [Python module](https://docs.inventree.org/en/latest/api/python/)
*   [Plugin interface](https://docs.inventree.org/en/latest/plugins/)
*   [Third party tools](https://docs.inventree.org/en/latest/plugins/integrate/)

<!-- TechStack -->
### :space_invader: Tech Stack

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
## :toolbox: Deployment / Getting Started

Deploying InvenTree is easy with multiple options:

<div align="center">
  <a href="https://docs.inventree.org/en/latest/start/docker/">Docker</a>
  <span> · </span>
  <a href="https://inventree.org/digitalocean"><img src="https://www.deploytodo.com/do-btn-blue-ghost.svg" alt="Deploy to DO" width="auto" height="40" /></a>
  <span> · </span>
  <a href="https://docs.inventree.org/en/latest/start/install/">Bare Metal</a>
</div>

**Quick Installation:**

For a single-line install, run the following command (refer to the [documentation](https://docs.inventree.org/en/latest/start/installer/) for supported distributions and details):

```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

For detailed instructions, see the [getting started guide](https://docs.inventree.org/en/latest/start/install/).

<!-- Mobile App -->
## :iphone: Mobile App

Enhance your InvenTree experience with the [companion mobile app](https://docs.inventree.org/app/), available on:

<div align="center">
  <a href="https://play.google.com/store/apps/details?id=inventree.inventree_app">Android Play Store</a>
  <span> · </span>
  <a href="https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone">Apple App Store</a>
</div>

<!-- Security -->
## :lock: Security

The InvenTree project prioritizes security, adhering to industry best practices.
*   Read our [Code of Conduct](CODE_OF_CONDUCT.md) for a welcoming environment.
*   Review our [Security Policy](SECURITY.md) for detailed security measures.
*   Find dedicated security information on [our documentation site](https://docs.inventree.org/en/latest/security/).

<!-- Contributing -->
## :wave: Contributing

We welcome and encourage contributions from the community!  Refer to the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/) for details.

<!-- Translation -->
## :scroll: Translation

Help translate the InvenTree web application via [Crowdin](https://crowdin.com/project/inventree)! **Contributions are highly encouraged.**

<!-- Sponsor -->
## :money_with_wings: Sponsor

Support the ongoing development of InvenTree by [sponsoring the project](https://github.com/sponsors/inventree).

<!-- Acknowledgments -->
## :gem: Acknowledgments

We acknowledge [PartKeepr](https://github.com/partkeepr/PartKeepr) as an inspiration.  Find a full list of used third-party libraries in the license information dialog of your instance.

## :heart: Support

<p>This project is supported by the following sponsors:</p>

<p align="center">
<a href="https://github.com/MartinLoeper"><img src="https://github.com/MartinLoeper.png" width="60px" alt="Martin Löper" /></a>
<a href="https://github.com/lippoliv"><img src="https://github.com/lippoliv.png" width="60px" alt="Oliver Lippert" /></a>
<a href="https://github.com/lfg-seth"><img src="https://github.com/lfg-seth.png" width="60px" alt="Seth Smith" /></a>
<a href="https://github.com/snorkrat"><img src="" width="60px" alt="" /></a>
<a href="https://github.com/spacequest-ltd"><img src="https://github.com/spacequest-ltd.png" width="60px" alt="SpaceQuest Ltd" /></a>
<a href="https://github.com/appwrite"><img src="https://github.com/appwrite.png" width="60px" alt="Appwrite" /></a>
<a href="https://github.com/PricelessToolkit"><img src="" width="60px" alt="" /></a>
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
Key improvements and explanations:

*   **SEO Optimization:**
    *   Added the target keyword "Inventory Management System" in the title and description.
    *   Included more descriptive feature bullet points.
    *   Used relevant headings.
    *   Added clear descriptions.
*   **Structure and Readability:**
    *   Consistent use of headings, subheadings, and bullet points for easy scanning.
    *   Grouped similar information (e.g., deployment options).
    *   Used emojis to visually break up sections.
*   **Conciseness:**
    *   Streamlined descriptions while retaining essential information.
    *   Removed redundant phrasing.
*   **Call to Action:** Added links to demo, docs and GitHub repo.
*   **Comprehensive Coverage:** Included all original sections, reformatting for clarity.
*   **Removed non-essential badges:** Kept the important badges.
*   **Improved introduction**:
    *   Added a hook sentence to grab the readers attention.
*   **Clear Language:** Used plain and concise language to make it easy to understand.