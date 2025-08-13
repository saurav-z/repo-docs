# Paperless-ngx: The Open-Source Document Management System

**Tired of paper clutter? Paperless-ngx is the document management system that digitizes your paperwork, making it searchable and accessible.**

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

Paperless-ngx is the official successor to the original [Paperless](https://github.com/the-paperless-project/paperless) & [Paperless-ng](https://github.com/jonaswinkler/paperless-ng) projects, designed to distribute the responsibility of advancing and supporting the project among a team of people.

**Key Features:**

*   **Document Scanning and Import:** Digitize documents from scanners, email, or import existing files.
*   **Optical Character Recognition (OCR):** Automatically convert scanned documents into searchable text.
*   **Automated Tagging and Organization:** Configure rules to automatically tag and categorize your documents.
*   **Full-Text Search:** Quickly find documents using keywords, tags, or dates.
*   **Web-Based Interface:** Access your documents from anywhere with a web browser.
*   **User Management:** Control access to your documents with user accounts and permissions.
*   **Easy Migration:** Seamlessly migrate from Paperless-ng.

[See the full list of features](https://docs.paperless-ngx.com/#features) and [screenshots](https://docs.paperless-ngx.com/#screenshots).

## Getting Started

The easiest way to deploy paperless is using `docker compose`. You can configure a `docker compose` environment with our install script:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

More details and step-by-step guides for alternative installation methods can be found in [the documentation](https://docs.paperless-ngx.com/setup/#installation).

## Documentation

Comprehensive documentation is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

We welcome contributions from the community! Please see the [CONTRIBUTING.md](https://github.com/paperless-ngx/paperless-ngx/blob/main/CONTRIBUTING.md) file for details.

### Community Support

Join the [Matrix Room](https://matrix.to/#/%23paperless:matrix.org) to connect with other users and contributors, or reach out through the [GitHub discussions](https://github.com/paperless-ngx/paperless-ngx/discussions).

### Translation

Help translate Paperless-ngx into your language on [Crowdin](https://crowdin.com/project/paperless-ngx).

### Feature Requests

Submit feature requests via [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).

### Bug Reports

Report bugs by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues) or starting a discussion.

## Related Projects

Check out the [wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects) for a list of projects that integrate with Paperless-ngx.

## Important Security Note

> **Important: Due to the sensitive nature of the documents stored, Paperless-ngx should be run on a trusted, secure server (e.g., your home server).** No guarantees are made regarding security (but we do try!) and you use the app at your own risk.
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

**[Back to the Paperless-ngx GitHub Repository](https://github.com/paperless-ngx/paperless-ngx)**