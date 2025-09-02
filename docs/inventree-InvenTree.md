<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
  <h1>InvenTree: Open-Source Inventory Management</h1>
  <p>Effortlessly track parts and manage your stock with InvenTree, a powerful and open-source inventory management system.</p>

<!-- Badges -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/inventree/inventree)]
[![CI](https://github.com/inventree/inventree/actions/workflows/qc_checks.yaml/badge.svg)]
[![Documentation Status](https://readthedocs.org/projects/inventree/badge/?version=latest)]
[![Docker Build](https://github.com/inventree/inventree/actions/workflows/docker.yaml/badge.svg)]
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
## :star2: About InvenTree

InvenTree is a powerful and flexible open-source inventory management system designed for tracking parts, managing stock, and streamlining your operations.  Built with a robust Python/Django backend, InvenTree offers a user-friendly web interface and a comprehensive REST API.  Extend its functionality with a powerful plugin system to customize InvenTree to your specific needs.  Learn more and explore the possibilities at the [InvenTree website](https://inventree.org).

**Key Features:**

*   **Web-Based Interface:** Intuitive interface for easy inventory management.
*   **REST API:** Integrate with other applications and systems.
*   **Part Tracking:** Detailed tracking of parts and components.
*   **Stock Control:** Efficient management of stock levels and locations.
*   **Plugin System:** Customize InvenTree with custom applications and extensions.
*   **Mobile App:** Access your inventory on the go with our companion mobile app.

<!-- Roadmap -->
### :compass: Roadmap & Development

Stay updated on what's coming next.  Explore the [roadmap tag](https://github.com/inventree/InvenTree/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap) and [horizon milestone](https://github.com/inventree/InvenTree/milestone/42) to follow our progress.

<!-- Integration -->
### :hammer_and_wrench: Integration & Extensibility

InvenTree is designed for seamless integration and extensibility.  Connect with external applications or develop custom plugins using these options:

*   [InvenTree API](https://docs.inventree.org/en/latest/api/)
*   [Python module](https://docs.inventree.org/en/latest/api/python/)
*   [Plugin interface](https://docs.inventree.org/en/latest/plugins/)
*   [Third party tools](https://docs.inventree.org/en/latest/plugins/integrate/)

<!-- TechStack -->
### :space_invader: Technology Stack

InvenTree is built using a modern and robust technology stack:

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
## 	:toolbox: Getting Started / Deployment

InvenTree offers multiple deployment options, making it easy to get started:

*   [Docker](https://docs.inventree.org/en/latest/start/docker/)
*   <a href="https://inventree.org/digitalocean"><img src="https://www.deploytodo.com/do-btn-blue-ghost.svg" alt="Deploy to DO" width="auto" height="40" /></a>
*   [Bare Metal](https://docs.inventree.org/en/latest/start/install/)

**Quick Installation:** Use the following one-line command (refer to the [docs](https://docs.inventree.org/en/latest/start/installer/) for supported distros and details):

```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

For detailed installation and setup instructions, consult the [getting started guide](https://docs.inventree.org/en/latest/start/install/).

<!-- Mobile App -->
## 	:iphone: Mobile App

Enhance your inventory management with the InvenTree companion mobile app, available for both Android and iOS:

<div align="center"><h4>
    <a href="https://play.google.com/store/apps/details?id=inventree.inventree_app">Android Play Store</a>
     <span> · </span>
    <a href="https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone">Apple App Store</a>
</h4></div>

<!-- Security -->
## :lock: Security & Code of Conduct

The InvenTree project prioritizes security and a welcoming community environment.

*   Review our [Code of Conduct](CODE_OF_CONDUCT.md).
*   Understand our [Security Policy](SECURITY.md).
*   Access dedicated security information on our [documentation site](https://docs.inventree.org/en/latest/security/).

<!-- Contributing -->
## :wave: Contributing

Join the InvenTree community! Your contributions are greatly appreciated.  See the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/) for guidance.

<!-- Translation -->
## :scroll: Translation

Help translate the InvenTree web application into your native language through [Crowdin](https://crowdin.com/project/inventree). **Your contributions are welcome!**

<!-- Sponsor -->
## :money_with_wings: Sponsor

Support InvenTree's development and help us improve the project! Consider [sponsoring the project](https://github.com/sponsors/inventree).

<!-- Acknowledgments -->
## :gem: Acknowledgements

We acknowledge [PartKeepr](https://github.com/partkeepr/PartKeepr) as a valuable predecessor and source of inspiration.  Find a complete list of third-party libraries in the license information dialog within your InvenTree instance.

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

InvenTree is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License.  View [LICENSE.txt](https://github.com/inventree/InvenTree/blob/master/LICENSE) for details.

<!-- Link to the original repo -->
<p>  <a href="https://github.com/inventree/InvenTree">  View the InvenTree repository on GitHub.</a></p>
```
Key improvements and explanations:

*   **SEO Optimization:**
    *   Added relevant keywords in the title and throughout the text (e.g., "Inventory Management," "Open-Source," "Part Tracking," "Stock Control").
    *   Focused the introductory sentence on a key benefit (e.g., "Effortlessly track parts and manage your stock...").
    *   Used headings effectively to structure the information, increasing readability and SEO value.
    *   Added descriptive text for key features.
    *   Included a call to action encouraging users to learn more.
*   **Structure and Readability:**
    *   Improved the use of headings and subheadings for better organization.
    *   Used bullet points for key features to enhance readability.
    *   Maintained the use of icons for visual appeal, improving user engagement.
*   **Content Clarity:**
    *   Simplified and clarified the language for better understanding.
    *   Combined redundant sections.
*   **Comprehensive Summary:**
    *   Expanded on the "About" section to be more informative, giving users a better understanding of the project's value proposition.
    *   More clearly explained the benefit of features.
*   **Call to Action:**
    *   Added a clear call to action with a link back to the original repository.
    *   Added website link in the about section.
*   **Consolidated Information:**
    *   Grouped related sections (e.g., deployment options under "Getting Started").
*   **Removed Redundancy:** streamlined the content where possible.
*   **Formatting:** Kept all the badges and the structure, and tried to integrate them into the surrounding text where possible.

This revised README is more informative, user-friendly, and SEO-optimized, making it easier for users to understand and engage with the InvenTree project.