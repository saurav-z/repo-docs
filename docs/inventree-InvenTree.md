<!-- InvenTree Logo and Title -->
<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
  <h1>InvenTree: Open Source Inventory Management</h1>
</div>

<!-- Badges and Links -->
<div align="center">
  <p>
    <a href="https://github.com/inventree/InvenTree">
      <img src="https://img.shields.io/github/stars/inventree/InvenTree?style=social" alt="GitHub stars">
    </a>
    <a href="https://twitter.com/inventreedb">
      <img src="https://img.shields.io/twitter/follow/inventreedb?style=social" alt="Twitter">
    </a>
    <a href="https://www.reddit.com/r/InvenTree/">
      <img src="https://img.shields.io/reddit/subreddit-subscribers/inventree?style=social" alt="Reddit">
    </a>
  </p>
  <p>
    <a href="https://demo.inventree.org/">View Demo</a> |
    <a href="https://docs.inventree.org/en/latest/">Documentation</a> |
    <a href="https://github.com/inventree/InvenTree/issues/new?template=bug_report.md&title=[BUG]">Report Bug</a> |
    <a href="https://github.com/inventree/InvenTree/issues/new?template=feature_request.md&title=[FR]">Request Feature</a>
  </p>
</div>

## 🚀 InvenTree: Your Comprehensive, Open-Source Solution for Inventory Management

InvenTree is a powerful, open-source inventory management system, perfect for tracking parts, managing stock, and streamlining your supply chain. [Explore the InvenTree repository](https://github.com/inventree/InvenTree) to get started!

<!-- Key Features -->
## ✨ Key Features

*   **Web-Based Interface:** Access your inventory from anywhere with a web browser.
*   **Part Tracking:** Detailed part information, including data sheets, images, and lifecycle management.
*   **Stock Control:** Manage stock levels, locations, and movements with ease.
*   **BOM Management:** Create and manage Bill of Materials (BOMs) for your products.
*   **REST API:** Integrate InvenTree with other systems and applications.
*   **Plugin System:** Extend InvenTree's functionality with custom plugins.
*   **Mobile App:** Companion mobile app for on-the-go access and stock control.
*   **Multi-Platform:** Runs on Docker, bare metal, and cloud platforms.

<!-- About the Project -->
## 💡 About the Project

InvenTree is a robust, open-source inventory management system designed to provide powerful low-level stock control and part tracking. Built with a Python/Django backend and a REST API, InvenTree offers a user-friendly web-based interface and a flexible plugin system for customization. Visit [our website](https://inventree.org) for more details.

<!-- Roadmap -->
### 🧭 Roadmap

Stay updated on our development progress by checking the [roadmap tag](https://github.com/inventree/InvenTree/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap) and the [horizon milestone](https://github.com/inventree/InvenTree/milestone/42).

<!-- Integration -->
### 🛠️ Integration

InvenTree is designed for seamless integration and extensibility:

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
## 🚀 Deployment / Getting Started

Deploy InvenTree with these options:

<div align="center">
    <a href="https://docs.inventree.org/en/latest/start/docker/">Docker</a>
    <span> · </span>
    <a href="https://inventree.org/digitalocean"><img src="https://www.deploytodo.com/do-btn-blue-ghost.svg" alt="Deploy to DO" width="auto" height="40" /></a>
    <span> · </span>
    <a href="https://docs.inventree.org/en/latest/start/install/">Bare Metal</a>
</div>

Quick install:

```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

For full installation instructions, refer to the [getting started guide](https://docs.inventree.org/en/latest/start/install/).

<!-- Mobile App -->
## 📱 Mobile App

Enhance your InvenTree experience with the companion mobile app:

<div align="center">
    <a href="https://play.google.com/store/apps/details?id=inventree.inventree_app">Android Play Store</a>
     <span> · </span>
    <a href="https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone">Apple App Store</a>
</div>

<!-- Security -->
## 🔒 Code of Conduct & Security

*   Review our [Code of Conduct](CODE_OF_CONDUCT.md).
*   Read our [Security Policy](SECURITY.md) and [Security Documentation](https://docs.inventree.org/en/latest/security/)

<!-- Contributing -->
## 🤝 Contributing

Contributions are welcome! Review the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/) to get started.

<!-- Translation -->
## 🌐 Translation

Help translate InvenTree! Contributions are welcomed via [Crowdin](https://crowdin.com/project/inventree).

<!-- Sponsor -->
## 💖 Sponsor

Support the project: [Sponsor InvenTree](https://github.com/sponsors/inventree).

<!-- Acknowledgments -->
## 💎 Acknowledgements

We want to acknowledge [PartKeepr](https://github.com/partkeepr/PartKeepr) as a valuable predecessor and inspiration.
Find a full list of used third-party libraries in the license information dialog of your instance.

<!-- Support -->
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

InvenTree is licensed under the [MIT License](https://choosealicense.com/licenses/mit/). See [LICENSE.txt](https://github.com/inventree/InvenTree/blob/master/LICENSE) for details.