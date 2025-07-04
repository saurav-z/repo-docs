<!-- Logo and Title -->
<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
  <h1>InvenTree</h1>
  <p>Open Source Inventory Management System </p>

  <!-- Badges - Concise and relevant -->
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
  [![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/inventree/inventree)]
  [![Docker Build](https://github.com/inventree/InvenTree/actions/workflows/docker.yaml/badge.svg)]
  [![Documentation Status](https://readthedocs.org/projects/inventree/badge/?version=latest)](https://inventree.readthedocs.io/en/latest/?badge=latest)
  [![Netlify Status](https://api.netlify.com/api/v1/badges/9bbb2101-0a4d-41e7-ad56-b63fb6053094/deploy-status)](https://app.netlify.com/sites/inventree/deploys)
  [![Stars](https://img.shields.io/github/stars/inventree/InvenTree?style=social&label=Stars)](https://github.com/inventree/InvenTree/)

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

<!-- SEO-Optimized Summary -->
## InvenTree: Streamline Your Inventory with a Powerful Open-Source Solution

InvenTree is a free and open-source inventory management system designed to help you track and manage parts, stock levels, and more with ease.  This makes it perfect for businesses, hobbyists, and anyone needing robust inventory control.  Learn more on the [InvenTree GitHub repository](https://github.com/inventree/InvenTree).

<!-- Key Features -->
## Key Features

*   **Comprehensive Inventory Tracking:** Manage parts, stock, and locations efficiently.
*   **Web-Based Interface:** Access your inventory data from anywhere with a web browser.
*   **REST API:** Integrate with other applications and systems seamlessly.
*   **Plugin Support:** Extend functionality with custom plugins.
*   **Mobile App:**  Convenient mobile access for stock control and information.
*   **Open Source:** Completely free to use, modify, and distribute.

<!-- About the Project -->
## About InvenTree

InvenTree provides powerful low-level stock control and part tracking. The core of the InvenTree system is a Python/Django database backend which provides an admin interface (web-based) and a REST API for interaction with external interfaces and applications. A powerful plugin system provides support for custom applications and extensions.

<!-- Roadmap -->
### Roadmap

See what's planned and in development:
*   [Roadmap](https://github.com/inventree/InvenTree/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap)
*   [Horizon Milestone](https://github.com/inventree/InvenTree/milestone/42)

<!-- Integration -->
### Integration

InvenTree is designed to be **extensible**, and provides multiple options for **integration** with external applications or addition of custom plugins:

*   [InvenTree API](https://docs.inventree.org/en/latest/api/)
*   [Python module](https://docs.inventree.org/en/latest/api/python/)
*   [Plugin interface](https://docs.inventree.org/en/latest/plugins/)
*   [Third party tools](https://docs.inventree.org/en/latest/plugins/integrate/)

<!-- Tech Stack -->
### Tech Stack

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
## Deployment / Getting Started

Choose your preferred deployment method:

<div align="center">
  <h4>
    <a href="https://docs.inventree.org/en/latest/start/docker/">Docker</a>
    <span> · </span>
    <a href="https://inventree.org/digitalocean"><img src="https://www.deploytodo.com/do-btn-blue-ghost.svg" alt="Deploy to DO" width="auto" height="40" /></a>
    <span> · </span>
    <a href="https://docs.inventree.org/en/latest/start/install/">Bare Metal</a>
  </h4>
</div>

Quick Install (for supported distros):
```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

For detailed installation instructions, refer to the [getting started guide](https://docs.inventree.org/en/latest/start/install/).

<!-- Mobile App -->
## Mobile App

Access your inventory on the go with the InvenTree companion mobile app:

<div align="center">
  <h4>
    <a href="https://play.google.com/store/apps/details?id=inventree.inventree_app">Android Play Store</a>
     <span> · </span>
    <a href="https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone">Apple App Store</a>
  </h4>
</div>

<!-- Security -->
## Security & Code of Conduct

*   Our [Code of Conduct](CODE_OF_CONDUCT.md) ensures a welcoming environment for all users.
*   InvenTree follows industry best practices for security; see our [security policy](SECURITY.md) and [security documentation](https://docs.inventree.org/en/latest/security/).

<!-- Contributing -->
## Contributing

We welcome and encourage contributions!  Check out the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/) to learn how to get involved.

<!-- Translation -->
## Translation

Help translate the InvenTree web application via [Crowdin](https://crowdin.com/project/inventree).  Your contributions are greatly appreciated.

<!-- Sponsor -->
## Sponsor

If you find InvenTree valuable, please consider [sponsoring the project](https://github.com/sponsors/inventree).

<!-- Acknowledgments -->
## Acknowledgements

We acknowledge [PartKeepr](https://github.com/partkeepr/PartKeepr) as a valuable predecessor and inspiration. Find a full list of used third-party libraries in the license information dialog of your instance.

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

<!-- License -->
## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See [LICENSE.txt](https://github.com/inventree/InvenTree/blob/master/LICENSE) for more information.
```
Key improvements and explanations:

*   **SEO Optimization:** The summary and key features are crafted with relevant keywords (Inventory Management, Open Source, Parts Tracking, Stock Control) to improve search engine visibility. Headings are used effectively.
*   **Concise Hook:** The one-sentence hook clearly states the core function and value proposition.
*   **Clear Structure:** The README is logically organized with distinct headings, making it easy to scan and understand.
*   **Bulleted Key Features:** Key features are presented in a bulleted list for easy readability.
*   **Link Back to Repo:**  The first sentence includes a direct link back to the original repository.
*   **Badge Optimization:** Badges are included but reduced to the most relevant ones (license, Docker Build, documentation status, deploy status, stars).
*   **Concise and Focused Content:**  Unnecessary fluff has been removed, and the content is kept concise and to the point.
*   **Mobile App Highlight:** The mobile app is emphasized.
*   **Call to Action:**  Clear calls to action (e.g., "View Demo," "Report Bug") are present.
*   **Contributor Guidance:**  The "Contributing" section emphasizes the welcome and encouragement to contribute.
*   **Updated Sponsor Section:**  The sponsors' section is kept, as it is important information.