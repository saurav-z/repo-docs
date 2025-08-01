<div align="center">
  <img src="assets/images/logo/inventree.png" alt="InvenTree logo" width="200" height="auto" />
  <h1>InvenTree</h1>
  <p>Open Source Inventory Management System </p>
</div>

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

## InvenTree: Your Open-Source Solution for Powerful Inventory Management

InvenTree is a comprehensive open-source inventory management system designed to provide robust stock control and part tracking capabilities, empowering businesses to efficiently manage their assets.  Visit the [InvenTree repository](https://github.com/inventree/InvenTree/) for more information.

**Key Features:**

*   **Comprehensive Inventory Tracking:** Easily manage and track parts, components, and stock levels.
*   **Web-Based Admin Interface:** User-friendly, web-based admin interface for easy access and control.
*   **REST API:** Integrate with external systems and applications using the REST API.
*   **Customization:**  Extend functionality with a powerful plugin system.
*   **Mobile App:** Access stock control information and functionality on the go with the companion mobile app.
*   **Extensible:** Easily integrate with external applications using the API, Python module, or plugin interface.
*   **Multi-Platform Support:** Offers various deployment options including Docker, bare metal, and cloud deployments.
*   **Community-Driven:** Benefit from community contributions and support through translations and sponsorship opportunities.

### Getting Started

Deploy InvenTree with these options:

*   [Docker](https://docs.inventree.org/en/latest/start/docker/)
*   [Deploy to Digital Ocean](https://inventree.org/digitalocean)
*   [Bare Metal Installation](https://docs.inventree.org/en/latest/start/install/)

Install with a single line:
```bash
wget -qO install.sh https://get.inventree.org && bash install.sh
```

### Mobile App

Access your inventory data on the go with the InvenTree mobile app:

*   [Android Play Store](https://play.google.com/store/apps/details?id=inventree.inventree_app)
*   [Apple App Store](https://apps.apple.com/au/app/inventree/id1581731101#?platform=iphone)

### Integration

InvenTree is designed to be easily integrated and extended:

*   [InvenTree API](https://docs.inventree.org/en/latest/api/)
*   [Python module](https://docs.inventree.org/en/latest/api/python/)
*   [Plugin interface](https://docs.inventree.org/en/latest/plugins/)
*   [Third party tools](https://docs.inventree.org/en/latest/plugins/integrate/)

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

###  Security

The InvenTree project prioritizes security. Read our [Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md), and find dedicated security pages on the [documentation site](https://docs.inventree.org/en/latest/security/).

###  Contributing

Contributions are welcome! Review the [contribution page](https://docs.inventree.org/en/latest/develop/contributing/).

###  Translation

Contribute to the translation of the InvenTree web application via [crowdin](https://crowdin.com/project/inventree).

### Sponsor

Support the project by [sponsoring](https://github.com/sponsors/inventree).

### Acknowledgements

Inspired by [PartKeepr](https://github.com/partkeepr/PartKeepr). See the license dialog for a list of used libraries.

###  Support

This project is supported by the following sponsors (list of sponsor images).

With ongoing resources provided by (list of resource images)

### License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See [LICENSE.txt](https://github.com/inventree/InvenTree/blob/master/LICENSE) for details.
```
Key improvements and optimizations:

*   **SEO-Friendly Title and Introduction:**  Uses the target keywords "Inventory Management System" and "Open Source" prominently, then quickly tells the user what the project is in a concise, search-engine-friendly way.
*   **Clear Headings:** Uses `<h1>` and `<h2>` tags and `###` for better readability and organization, which is good for both users and SEO.
*   **Bulleted Key Features:**  Clearly highlights the main benefits of InvenTree, making it easy for users to understand its capabilities at a glance.
*   **Direct Links to Key Resources:** Provides easy access to the demo, documentation, and issue reporting.
*   **Deployment Section:** Adds a dedicated section with easy links to Docker and bare metal installs.
*   **Mobile App Section:** Adds a dedicated section to show the mobile app store links.
*   **Contribution and Sponsorship sections:**  These sections promote community involvement.
*   **Clear Licensing Information:**  Includes a license section.
*   **Code of Conduct and Security Sections**: Addresses security directly to increase user trust.
*   **Concise Language:** Uses clear and direct language throughout.
*   **Removed redundant badges**: Removed redundant badges for brevity