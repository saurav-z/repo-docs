# Paperless-ngx: Your Digital Document Management System

**Tired of paper clutter? Paperless-ngx is a powerful, open-source document management system that helps you effortlessly organize, archive, and search your documents.**  [Explore the original repository](https://github.com/paperless-ngx/paperless-ngx).

[![CI](https://github.com/paperless-ngx/paperless-ngx/workflows/ci/badge.svg)](https://github.com/paperless-ngx/paperless-ngx/actions)
[![Crowdin](https://badges.crowdin.net/paperless-ngx/localized.svg)](https://crowdin.com/project/paperless-ngx)
[![Documentation Status](https://img.shields.io/github/deployments/paperless-ngx/paperless-ngx/github-pages?label=docs)](https://docs.paperless-ngx.com)
[![Codecov](https://codecov.io/gh/paperless-ngx/paperless-ngx/branch/main/graph/badge.svg?token=VK6OUPJ3TY)](https://codecov.io/gh/paperless-ngx/paperless-ngx)
[![Chat on Matrix](https://matrix.to/img/matrix-badge.svg)](https://matrix.to/#/%23paperlessngx%3Amatrix.org)
[![Demo](https://cronitor.io/badges/ve7ItY/production/W5E_B9jkelG9ZbDiNHUPQEVH3MY.svg)](https://demo.paperless-ngx.com)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/paperless-ngx/paperless-ngx/blob/main/resources/logo/web/png/White%20logo%20-%20no%20background.png" width="50%">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/paperless-ngx/paperless-ngx/raw/main/resources/logo/web/png/Black%20logo%20-%20no%20background.png" width="50%">
    <img src="https://github.com/paperless-ngx/paperless-ngx/raw/main/resources/logo/web/png/Black%20logo%20-%20no%20background.png" width="50%">
  </picture>
</p>

Paperless-ngx is the official successor to Paperless and Paperless-ng, built to provide a robust, user-friendly solution for managing your documents digitally.

## Key Features

*   **OCR (Optical Character Recognition):** Convert scanned documents into searchable text.
*   **Automated Tagging and Organization:**  Intelligent document classification and tagging for effortless retrieval.
*   **Full-Text Search:** Quickly find documents using keywords and phrases.
*   **User-Friendly Interface:**  Intuitive design for easy navigation and management.
*   **Web-Based Access:** Access your documents from anywhere, anytime.
*   **Open Source:**  Free to use and customize, with a thriving community.
*   **Docker Compose Support:** Easy setup with Docker Compose for quick deployment.
*   **Migration from Paperless-ng:** Easily migrate existing documents.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards.png">
  <img src="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards.png">
</picture>

## Getting Started

The easiest way to get started is with Docker Compose:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

Detailed installation guides and alternative methods are available in the [documentation](https://docs.paperless-ngx.com/setup/#installation).

## Documentation

Comprehensive documentation is available at: [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contribute

We welcome contributions!  Help improve Paperless-ngx through bug fixes, feature enhancements, and more.  Check the [documentation](https://docs.paperless-ngx.com/development/) for information on getting started.

### Community Support

Join our community on GitHub and in the [Matrix Room](https://matrix.to/#/#paperless:matrix.org) to connect with other users and contributors.  We have multiple teams available for ongoing contributions.

### Translation

Help translate Paperless-ngx into your language on Crowdin: [https://crowdin.com/project/paperless-ngx](https://crowdin.com/project/paperless-ngx).

### Feature Requests

Submit feature requests and discuss new ideas on [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).

### Bugs

Report bugs or ask questions by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues) or starting a discussion on [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions).

## Related Projects

Explore related projects and integrations on the [wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects).

## Important Security Note

> **Important:**  Paperless-ngx is designed for use on trusted hosts. Data is stored in clear text without encryption.  We recommend running Paperless-ngx on a local server with backups.