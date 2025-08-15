# Paperless-ngx: Your Open-Source Document Management System

**Tired of paper clutter?** Paperless-ngx is a powerful, open-source document management system that helps you digitize, organize, and access your documents with ease.  Learn more and contribute at the [original repository](https://github.com/paperless-ngx/paperless-ngx).

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

## Key Features

*   **Document Digitization:** Scan and import documents easily.
*   **OCR & Full-Text Search:** Optical Character Recognition (OCR) enables searching within documents.
*   **Automated Tagging & Consumption:** Organize documents with automated processes.
*   **User-Friendly Interface:** Intuitive web-based interface for easy document management.
*   **Open Source:** Benefit from community contributions, transparency, and customization.
*   **Demo Available:** Try out the functionality with the demo at [demo.paperless-ngx.com](https://demo.paperless-ngx.com) (login `demo` / `demo`).

## Getting Started

The easiest way to deploy paperless is using `docker compose`. Preconfigured files are available in the [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose).

Quick start using install script:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

Find detailed installation guides in the [documentation](https://docs.paperless-ngx.com/setup/#installation).

## Documentation

Comprehensive documentation is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

We welcome contributions!  Please see the [documentation](https://docs.paperless-ngx.com/development/) for information on getting started.

### Community Support

*   Join the community on the [Matrix Room](https://matrix.to/#/#paperless:matrix.org) or engage on Github.
*   Consider contributing to the [project teams](https://github.com/orgs/paperless-ngx/people) (frontend, ci/cd, etc).

### Translation

Help translate Paperless-ngx into your language on [Crowdin](https://crowdin.com/project/paperless-ngx).  More details in [CONTRIBUTING.md](https://github.com/paperless-ngx/paperless-ngx/blob/main/CONTRIBUTING.md#translating-paperless-ngx).

### Feature Requests

Submit and discuss feature requests via [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).

### Bugs

Report bugs or ask questions by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues) or starting a [discussion](https://github.com/paperless-ngx/paperless-ngx/discussions).

## Related Projects

See the [wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects) for a user-maintained list of related projects and software.

## Important Security Note

> **Important Security Note:** Paperless-ngx should *never* be run on an untrusted host. Protect your data by running Paperless-ngx on a local server with backups, as information is stored in clear text without encryption.  Use at your own risk.
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