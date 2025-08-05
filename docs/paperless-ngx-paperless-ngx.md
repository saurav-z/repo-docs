[![ci](https://github.com/paperless-ngx/paperless-ngx/workflows/ci/badge.svg)](https://github.com/paperless-ngx/paperless-ngx/actions)
[![Crowdin](https://badges.crowdin.net/paperless-ngx/localized.svg)](https://crowdin.com/project/paperless-ngx)
[![Documentation Status](https://img.shields.io/github/deployments/paperless-ngx/paperless-ngx/github-pages?label=docs)](https://docs.paperless-ngx.com)
[![codecov](https://codecov.io/gh/paperless-ngx/paperless-ngx/branch/main/graph/badge.svg?token=VK6OUPJ3TY)](https://codecov.io/gh/paperless-ngx/paperless-ngx)
[![Chat on Matrix](https://matrix.to/img/matrix-badge.svg)](https://matrix.to/#/%23paperlessngx%3Amatrix.org)
[![demo](https://cronitor.io/badges/ve7ItY/production/W5E_B9jkelG9ZbDiNHUPQEVH3MY.svg)](https://demo.paperless-ngx.com)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/paperless-ngx/paperless-ngx/blob/main/resources/logo/web/png/White%20logo%20-%20no%20background.png" width="50%">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/paperless-ngx/paperless-ngx/raw/main/resources/logo/web/png/Black%20logo%20-%20no%20background.png" width="50%">
    <img src="https://github.com/paperless-ngx/paperless-ngx/raw/main/resources/logo/web/png/Black%20logo%20-%20no%20background.png" width="50%">
  </picture>
</p>

## Paperless-ngx: Organize Your Documents Digitally and Declutter Your Life

Paperless-ngx is an open-source document management system that helps you scan, organize, and archive your documents in a searchable digital format.  [Explore the Paperless-ngx repository on GitHub](https://github.com/paperless-ngx/paperless-ngx)!

**Key Features:**

*   **Easy Document Upload:** Quickly import documents via scanning, drag-and-drop, or email.
*   **Optical Character Recognition (OCR):** Automatically extract text from scanned documents, making them fully searchable.
*   **Automated Tagging and Indexing:**  Intelligent features to automatically categorize and organize your documents.
*   **Powerful Search:** Find documents instantly with advanced search capabilities, including full-text search.
*   **User-Friendly Interface:**  Intuitive web interface for easy access and management of your documents.
*   **Open Source & Self-Hosted:**  Take control of your documents and data by hosting Paperless-ngx yourself.
*   **Mobile-friendly**: Easy to access documents on the go.

    
    
    
    
    
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards.png">
  <img src="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards.png">
</picture>

For a full list of [features](https://docs.paperless-ngx.com/#features) and [screenshots](https://docs.paperless-ngx.com/#screenshots), please refer to the [documentation](https://docs.paperless-ngx.com/).

### Getting Started

The easiest way to deploy Paperless-ngx is using `docker compose`.  Refer to the files in the [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose) for pre-configured examples that pull images from the GitHub container registry.

Alternatively, you can configure a `docker compose` environment using the install script:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

Detailed installation guides for various methods are available in the [documentation](https://docs.paperless-ngx.com/setup/#installation).

Users of Paperless-ng can easily migrate by simply using the new docker image.  See the [documentation on migrating](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx) for instructions.

### Documentation

Comprehensive documentation for Paperless-ngx is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

### Contributing

We welcome contributions to the project! Bug fixes, enhancements, and visual improvements are all appreciated.  For significant features, please start a discussion beforehand. The [documentation](https://docs.paperless-ngx.com/development/) provides basic information on how to get started.

#### Community Support

Join the Paperless-ngx community on GitHub and in the [Matrix Room](https://matrix.to/#/#paperless:matrix.org) to connect with other users and contributors.  If you're interested in ongoing project involvement, consider joining one of the teams (frontend, CI/CD, etc.) listed on the [organization's people page](https://github.com/orgs/paperless-ngx/people).

#### Translation

Help translate Paperless-ngx into your language via [Crowdin](https://crowdin.com/project/paperless-ngx).  More details can be found in [CONTRIBUTING.md](https://github.com/paperless-ngx/paperless-ngx/blob/main/CONTRIBUTING.md#translating-paperless-ngx).

#### Feature Requests

Submit feature requests and discuss ideas on [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).

#### Bugs

Report bugs by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues) or initiating a discussion on GitHub.

### Related Projects

Find a user-maintained list of related projects and compatible software on the [wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects).

### Important Note: Security Considerations

> When running Paperless-ngx, always prioritize security. Never run it on an untrusted host.  Information is stored in clear text without encryption, so the safest way to use Paperless-ngx is on a local server in your home with regular backups.  Use at your own risk; the project makes no guarantees regarding security, though it is a priority.