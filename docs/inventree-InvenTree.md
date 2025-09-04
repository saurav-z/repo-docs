<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
  <h1>InvenTree</h1>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/inventree/inventree)]
![CI](https://github.com/inventree/InvenTree/actions/workflows/qc_checks.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/inventree/badge/?version=latest)]
![Docker Build](https://github.com/inventree/InvenTree/actions/workflows/docker.yaml/badge.svg)
[![Netlify Status](https://api.netlify.com/api/v1/badges/9bbb2101-0a4d-41e7-ad56-b63fb6053094/deploy-status)]
[![Performance Testing](https://dev.azure.com/InvenTree/InvenTree%20test%20statistics/_apis/build/status%2Fmatmair.InvenTree?branchName=testing)]

[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/7179/badge)]
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/inventree/InvenTree/badge)]
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=inventree_InvenTree&metric=sqale_rating)]

[![codecov](https://codecov.io/gh/inventree/InvenTree/graph/badge.svg?token=9DZRGUUV7B)]
[![Crowdin](https://badges.crowdin.net/inventree/localized.svg)]
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/inventree/inventree)
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

## InvenTree: The Open-Source Inventory Management System for Modern Businesses

InvenTree is a powerful, open-source solution for managing your inventory and tracking parts, designed to streamline your operations and improve efficiency.  [Check out the InvenTree GitHub Repo](https://github.com/inventree/InvenTree).

**Key Features:**

*   **Comprehensive Inventory Tracking:**  Manage stock levels, locations, and part details with ease.
*   **Web-Based Admin Interface:** Access and manage your inventory from anywhere with a web browser.
*   **REST API & Integrations:** Integrate InvenTree with other systems using its robust REST API and plugin system.
*   **Extensible with Plugins:** Customize and extend InvenTree's functionality to meet your specific needs.
*   **Mobile App Support:**  Access your inventory on the go with the companion mobile app.

## Core Features

*   **Part Management:**
    *   Organize parts with detailed information.
    *   Manage part revisions and parameters.
    *   Track manufacturer and supplier information.
    *   Link parts to BOMs and stock.
*   **Stock Control:**
    *   Track stock quantities and locations.
    *   Manage stock transfers and adjustments.
    *   Generate and track purchase orders and sales orders.
*   **BOM Management:**
    *   Define Bill of Materials (BOMs) for your products.
    *   Track component consumption during assembly.
    *   Manage BOM revisions and variants.
*   **Reporting and Analytics:**
    *   Generate reports on stock levels, usage, and more.
    *   Gain insights into your inventory data.
*   **User Management:**
    *   Control user access and permissions.
    *   Manage user roles and groups.
*   **API and Integrations:**
    *   Integrate with other systems using the REST API.
    *   Extend functionality with plugins.

## Getting Started

InvenTree offers several deployment options:

*   **Docker:**  Deploy quickly and easily using Docker. ([Docker Documentation](https://docs.inventree.org/en/latest/start/docker/))
*   **Digital Ocean:** Deploy InvenTree with a one-click deploy to Digital Ocean.
*   **Bare Metal:** Install directly on your server. ([Bare Metal Installation](https://docs.inventree.org/en/latest/start/install/))

Use the following one-line command to install.  See the [full installation instructions](https://docs.inventree.org/en/latest/start/installer/) for details:
```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

## Mobile App

Get access to InvenTree from your mobile device with the companion mobile app:

*   [Android Play Store](https://play.google.com/store/apps/details?id=inventree.inventree_app)
*   [Apple App Store](https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone)

##  Integrations

*   **API:**  [InvenTree API](https://docs.inventree.org/en/latest/api/)
*   **Python Module:**  [Python module](https://docs.inventree.org/en/latest/api/python/)
*   **Plugin Interface:**  [Plugin interface](https://docs.inventree.org/en/latest/plugins/)
*   **Third party tools:**  [Third party tools](https://docs.inventree.org/en/latest/plugins/integrate/)

## Tech Stack

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


##  Roadmap

Stay updated on the development with the [roadmap tag](https://github.com/inventree/InvenTree/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap) and [horizon milestone](https://github.com/inventree/InvenTree/milestone/42).

## Security & Community

*   **Code of Conduct:** [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
*   **Security Policy:** [SECURITY.md](SECURITY.md)
*   **Documentation:** [Security Documentation](https://docs.inventree.org/en/latest/security/)

## Contributing

We welcome and encourage contributions from the community; please refer to the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/).

## Translation

Help translate the InvenTree web application into your native language. Contributions are welcome via [Crowdin](https://crowdin.com/project/inventree).

##  Sponsor

Consider [sponsoring the project](https://github.com/sponsors/inventree) if you find InvenTree useful.

## Acknowledgments

We acknowledge [PartKeepr](https://github.com/partkeepr/PartKeepr) as a valuable predecessor and source of inspiration.

## Support

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
```

Key improvements and summaries:

*   **SEO Optimization:**  Keywords like "inventory management system," "open source," and relevant features are included throughout the document.
*   **Clear Headings:**  Uses clear and concise headings for each section, improving readability and organization.
*   **Bulleted Key Features:**  Highlights the core features in a concise, easy-to-scan format.
*   **Concise Language:** The text is more direct and to the point.
*   **One-Sentence Hook:**  A compelling opening sentence to immediately grab the reader's attention.
*   **Actionable Links:**  Includes links to the demo, documentation, and the GitHub repository for easy access.
*   **Organization and Structure:** Sections are logically arranged, improving the overall flow.
*   **Removed Unnecessary Badges**: Cleaned up the badge section to remove redundancy.
*   **Updated Installation Instructions:** Added installation instructions.
*   **Added Core Feature bullet points:** Expanded on each of the core features.