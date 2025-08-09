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

**Paperless-ngx is a powerful, open-source document management system (DMS) designed to help you effortlessly organize, archive, and search your documents, helping you eliminate paper clutter.**  [Explore the original repo](https://github.com/paperless-ngx/paperless-ngx).

Paperless-ngx is the official successor to the original [Paperless](https://github.com/the-paperless-project/paperless) & [Paperless-ng](https://github.com/jonaswinkler/paperless-ng) projects.

**Key Features:**

*   **Automated Document Scanning and Import:** Easily import documents from various sources, including scanners, email, and cloud storage.
*   **Optical Character Recognition (OCR):**  Convert scanned documents into searchable text.
*   **Intelligent Tagging and Categorization:**  Automatically tag and categorize your documents based on customizable rules.
*   **Full-Text Search:** Quickly find any document using powerful search capabilities.
*   **Secure Storage:**  Organize and store documents securely with user-configurable settings.
*   **Web-Based Interface:**  Access your documents from any device with a web browser.
*   **REST API:** Interact and integrate Paperless-ngx with external services.
*   **Open Source and Self-Hosted:**  Take control of your data and manage it on your own server.

### Demo

Experience Paperless-ngx firsthand!  A demo is available at [demo.paperless-ngx.com](https://demo.paperless-ngx.com) with login `demo` / `demo`. _Note: demo content is reset frequently and confidential information should not be uploaded._

### Getting Started

The easiest way to deploy Paperless-ngx is using `docker compose`.

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

Detailed installation guides for other methods can be found in [the documentation](https://docs.paperless-ngx.com/setup/#installation).

Migrating from Paperless-ng is easy, just drop in the new docker image! See the [documentation on migrating](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx) for more details.

### Documentation

Comprehensive documentation is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

### Contributing

We welcome contributions!  Find information on how to get started in the [documentation](https://docs.paperless-ngx.com/development/).

*   **Community Support:** Connect with the Paperless-ngx community on the [Matrix Room](https://matrix.to/#/#paperless:matrix.org).
*   **Translation:** Help translate Paperless-ngx into your language via [Crowdin](https://crowdin.com/project/paperless-ngx).
*   **Feature Requests:** Suggest and discuss new features in [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).
*   **Bug Reporting:**  Report any bugs or issues via [GitHub Issues](https://github.com/paperless-ngx/paperless-ngx/issues).

### Related Projects

Explore compatible projects and integrations in the [wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects).

### Important Note - Security Considerations

> Document scanners are typically used to scan sensitive documents like your social insurance number, tax records, invoices, etc. **Paperless-ngx should never be run on an untrusted host** because information is stored in clear text without encryption. No guarantees are made regarding security (but we do try!) and you use the app at your own risk.
> **The safest way to run Paperless-ngx is on a local server in your own home with backups in place**.