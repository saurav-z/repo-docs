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

# Paperless-ngx: Your Digital Document Management Solution

**Paperless-ngx is a powerful, open-source document management system designed to help you declutter your life by turning paper documents into a searchable, accessible digital archive.** Learn more and contribute on [GitHub](https://github.com/paperless-ngx/paperless-ngx).

**Key Features:**

*   **Document Scanning and Import:** Easily scan or import documents from various sources.
*   **Optical Character Recognition (OCR):** Automatic text extraction for searchable documents.
*   **Automated Tagging and Organization:** Intelligent features to categorize and organize your documents.
*   **Full-Text Search:** Quickly find documents using keywords and phrases.
*   **Web-Based Interface:** Access your documents from anywhere with a web browser.
*   **User-Friendly Interface:**  Intuitive and easy-to-use interface for seamless document management.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards.png">
  <img src="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards.png">
</picture>

## Getting Started

The easiest way to deploy Paperless-ngx is using `docker compose`. The files in the [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose) are configured to pull the image from the GitHub container registry.

To quickly set up a `docker compose` environment, run the following installation script:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

For alternative installation methods and detailed step-by-step guides, see the [documentation](https://docs.paperless-ngx.com/setup/#installation).

Migrating from Paperless-ng is straightforward—just drop in the new Docker image! Refer to the [documentation](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx) for migration details.

## Documentation

Comprehensive documentation is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

We welcome contributions!  Whether it's bug fixes, enhancements, or visual improvements, your help is appreciated.

*   **Community Support:** Connect with other users and contributors on GitHub and in the [Matrix Room](https://matrix.to/#/#paperless:matrix.org). Explore the [teams](https://github.com/orgs/paperless-ngx/people) to see how you can help.
*   **Translation:** Help translate Paperless-ngx into your language at [Crowdin](https://crowdin.com/project/paperless-ngx).
*   **Feature Requests:** Submit your ideas and vote on existing ones in [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).
*   **Bugs:** Report bugs or ask questions by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues) or [starting a discussion](https://github.com/paperless-ngx/paperless-ngx/discussions).

## Related Projects

Find a user-maintained list of compatible projects and software on the [wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects).

## Important Security Note

> Document scanners are typically used to scan sensitive documents like your social insurance number, tax records, invoices, etc. **Paperless-ngx should never be run on an untrusted host** because information is stored in clear text without encryption. No guarantees are made regarding security (but we do try!) and you use the app at your own risk.
> **The safest way to run Paperless-ngx is on a local server in your own home with backups in place**.