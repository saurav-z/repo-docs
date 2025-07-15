<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
  <h1>InvenTree: Open Source Inventory Management</h1>

  <p>
     **InvenTree is your open-source solution for powerful, low-level stock control and efficient part tracking.**
  </p>

<!-- Badges -->
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

<!-- About the Project -->
## :star2: About InvenTree

InvenTree is a robust, open-source inventory management system ideal for businesses and individuals needing detailed stock control and component tracking. Built on a Python/Django backend, it offers a user-friendly web interface and a powerful REST API for seamless integration.  Extend its functionality through a comprehensive plugin system. Visit the [InvenTree website](https://inventree.org) to learn more.

**Key Features:**

*   **Comprehensive Inventory Management:** Track parts, components, and stock levels with ease.
*   **Web-Based Interface:**  Access and manage your inventory from any web browser.
*   **REST API:** Integrate with other applications and systems.
*   **Plugin System:** Customize and extend InvenTree's functionality.
*   **User-Friendly:** Intuitive interface for easy navigation and management.

### :compass: Roadmap

Explore the future of InvenTree with the [roadmap](https://github.com/inventree/InvenTree/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap) and [horizon milestone](https://github.com/inventree/InvenTree/milestone/42).

### :hammer_and_wrench: Integration & Extensibility

InvenTree is designed for seamless integration and customization:

*   [InvenTree API](https://docs.inventree.org/en/latest/api/)
*   [Python module](https://docs.inventree.org/en/latest/api/python/)
*   [Plugin interface](https://docs.inventree.org/en/latest/plugins/)
*   [Third party tools](https://docs.inventree.org/en/latest/plugins/integrate/)

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
## :toolbox: Getting Started & Deployment

Deploying InvenTree is simple with various options available.

<div align="center"><h4>
    <a href="https://docs.inventree.org/en/latest/start/docker/">Docker</a>
    <span> · </span>
    <a href="https://inventree.org/digitalocean"><img src="https://www.deploytodo.com/do-btn-blue-ghost.svg" alt="Deploy to DO" width="auto" height="40" /></a>
    <span> · </span>
    <a href="https://docs.inventree.org/en/latest/start/install/">Bare Metal</a>
</h4></div>

Quick Installation using:
```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

Follow the [getting started guide](https://docs.inventree.org/en/latest/start/install/) for comprehensive installation and setup instructions.

<!-- Mobile App -->
## :iphone: Mobile App

Manage your inventory on the go with the [InvenTree mobile app](https://docs.inventree.org/app/).

<div align="center"><h4>
    <a href="https://play.google.com/store/apps/details?id=inventree.inventree_app">Android Play Store</a>
     <span> · </span>
    <a href="https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone">Apple App Store</a>
</h4></div>

<!-- Security -->
## :lock: Security & Code of Conduct

The InvenTree project is committed to security and a welcoming community.  Review our [Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md). You can also find dedicated security information [on our documentation site](https://docs.inventree.org/en/latest/security/).

<!-- Contributing -->
## :wave: Contributing

We welcome contributions!  Review the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/) to get started.  Contribute to the open-source InvenTree project [here](https://github.com/inventree/InvenTree).

<!-- Translation -->
## :scroll: Translation

Help translate InvenTree into your language via [Crowdin](https://crowdin.com/project/inventree).

<!-- Sponsor -->
## :money_with_wings: Sponsor

Support the development of InvenTree by [sponsoring the project](https://github.com/sponsors/inventree).

<!-- Acknowledgments -->
## :gem: Acknowledgements

We appreciate the inspiration from [PartKeepr](https://github.com/partkeepr/PartKeepr).

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

*   **Clear Headline:**  Includes the core keyword "Inventory Management" in the H1.
*   **Concise Hook:**  The opening sentence is a compelling, benefit-driven statement.
*   **Keyword Optimization:**  Repeatedly uses relevant keywords like "inventory management," "open source," "stock control," and "part tracking".
*   **Bulleted Features:** Highlights key advantages for easy scanning.
*   **Structured Headings:** Organizes content for readability and SEO (H2s and H3s).
*   **Internal Links:**  Includes links to key sections (Roadmap, API, Mobile App, etc.) for better user experience.
*   **Call to Actions:** Encourages engagement (Contributing, Sponsoring).
*   **Link to Original Repo:**  Clear link back to the source.
*   **Emphasis on Benefits:** Focuses on what users *get* from InvenTree.
*   **Mobile App and Deployment Highlights:** Prominently displays these crucial aspects.
*   **SEO friendly titles**: Added title for each section.