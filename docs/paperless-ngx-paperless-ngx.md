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

<!-- omit in toc -->

# Paperless-ngx: Your Open-Source Document Management Solution

**Paperless-ngx is the document management system that helps you declutter your life by turning paper documents into a searchable digital archive.** [Visit the original repository](https://github.com/paperless-ngx/paperless-ngx).

Paperless-ngx is the official successor to the original [Paperless](https://github.com/the-paperless-project/paperless) & [Paperless-ng](https://github.com/jonaswinkler/paperless-ng) projects, built to provide a robust and community-driven solution for managing your documents.

A demo is available at [demo.paperless-ngx.com](https://demo.paperless-ngx.com) using login `demo` / `demo`. _Note: demo content is reset frequently and confidential information should not be uploaded._

## Key Features

*   **Automated Document Organization:** Automatically import, process, and organize your documents.
*   **Full-Text Search:** Easily search the content of your documents using OCR.
*   **Customizable Metadata:** Tag, categorize, and add custom metadata to your documents.
*   **Web-Based Interface:** Access your documents from anywhere with a web browser.
*   **OCR (Optical Character Recognition):** Extract text from scanned documents.
*   **Document Preview:** Quickly view documents in the browser.

A full list of [features](https://docs.paperless-ngx.com/#features) and [screenshots](https://docs.paperless-ngx.com/#screenshots) are available in the [documentation](https://docs.paperless-ngx.com/).

## Getting Started

The easiest way to deploy Paperless-ngx is using `docker compose`. The files in the [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose) are configured to pull the image from the GitHub container registry.

To quickly get started, you can use the install script:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

Detailed installation instructions for alternative methods are available in the [documentation](https://docs.paperless-ngx.com/setup/#installation).

Migrating from Paperless-ng is easy; simply use the new Docker image! See the [documentation on migrating](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx).

### Documentation

Comprehensive documentation is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

We welcome contributions! Bug fixes, enhancements, and visual improvements are always appreciated.  If you want to implement something significant, please start a discussion. Information on getting started is in the [documentation](https://docs.paperless-ngx.com/development/).

### Community Support

Join the community on GitHub and in the [Matrix Room](https://matrix.to/#/#paperless:matrix.org). If you'd like to contribute on an ongoing basis, explore the different [teams](https://github.com/orgs/paperless-ngx/people) (frontend, CI/CD, etc.) and offer your help!

### Translation

Paperless-ngx is available in many languages, coordinated on Crowdin.  Help translate Paperless-ngx into your language at https://crowdin.com/project/paperless-ngx. More details can be found in [CONTRIBUTING.md](https://github.com/paperless-ngx/paperless-ngx/blob/main/CONTRIBUTING.md#translating-paperless-ngx).

### Feature Requests

Submit feature requests via [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests). Search for existing ideas, add your own, and vote for the ones you care about.

### Bugs

Report bugs by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues) or [start a discussion](https://github.com/paperless-ngx/paperless-ngx/discussions).

## Related Projects

See [the wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects) for a user-maintained list of related projects and software compatible with Paperless-ngx.

## Important Note

> Document scanners are typically used to scan sensitive documents like your social insurance number, tax records, invoices, etc. **Paperless-ngx should never be run on an untrusted host** because information is stored in clear text without encryption. No guarantees are made regarding security (but we do try!) and you use the app at your own risk.
> **The safest way to run Paperless-ngx is on a local server in your own home with backups in place**.