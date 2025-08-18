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

## Paperless-ngx: Your Digital Document Management Solution

Paperless-ngx is a powerful, open-source document management system that helps you effortlessly organize, search, and archive your documents, freeing you from the clutter of paper.  [Visit the original repo for more information](https://github.com/paperless-ngx/paperless-ngx).

**Key Features:**

*   **Automated Document Processing:** Automatically imports, OCRs, and indexes your documents for easy searching.
*   **Full-Text Search:**  Find documents quickly with robust full-text search capabilities.
*   **Tagging and Categorization:** Organize documents with custom tags, document types, and correspondents.
*   **Web-Based Interface:** Access your documents from anywhere with a user-friendly web interface.
*   **Open Source:** Benefit from community contributions and full control over your data.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards.png">
  <img src="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards.png">
</picture>

**[Features and Screenshots](https://docs.paperless-ngx.com/#features) are available in the [documentation](https://docs.paperless-ngx.com/).**

## Getting Started

The easiest way to deploy Paperless-ngx is using `docker compose`. Configuration files are available in the [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose) and are configured to pull the image from the GitHub container registry.

**Quick Installation:**

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

**Migrating from Paperless-ng:**  Migrating from Paperless-ng is easy; simply use the new Docker image.  See the [documentation on migrating](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx) for more details.

### Documentation

Comprehensive documentation is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

We welcome contributions!  Help improve Paperless-ngx by contributing bug fixes, enhancements, or visual improvements.  For larger features, please start a discussion.

*   **[Community Support](https://matrix.to/#/#paperless:matrix.org):** Join the community on Matrix to discuss and contribute.
*   **[Translation](https://crowdin.com/project/paperless-ngx):** Help translate Paperless-ngx into your language on Crowdin.
*   **[Feature Requests](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests):** Submit and vote on feature requests via GitHub Discussions.
*   **[Bugs](https://github.com/paperless-ngx/paperless-ngx/issues):** Report bugs or ask questions by opening an issue.

## Related Projects

Explore related projects and compatible software on [the wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects).

## Important Note: Security

> Document scanners are typically used to scan sensitive documents. **Paperless-ngx should never be run on an untrusted host** because information is stored in clear text without encryption. No guarantees are made regarding security (but we do try!) and you use the app at your own risk.
> **The safest way to run Paperless-ngx is on a local server in your own home with backups in place**.

<p align="right">This project is supported by:<br/>
  <a href="https://m.do.co/c/8d70b916d462" style="padding-top: 4px; display: block;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_white.svg" width="140px">
      <source media="(prefers-color-scheme: light)" srcset="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_blue.svg" width="140px">
      <img src="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_black_.svg" width="140px">
    </picture>
  </a>
</p>