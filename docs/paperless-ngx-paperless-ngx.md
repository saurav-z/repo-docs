# Paperless-ngx: Effortlessly Organize and Search Your Documents

Paperless-ngx is a powerful, open-source document management system that transforms your physical documents into a searchable digital archive, making it easy to find and manage your important papers. **Digitize your documents and declutter your life with Paperless-ngx!**  Learn more and contribute at the [Paperless-ngx GitHub Repository](https://github.com/paperless-ngx/paperless-ngx).

[![CI](https://github.com/paperless-ngx/paperless-ngx/workflows/ci/badge.svg)](https://github.com/paperless-ngx/paperless-ngx/actions)
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

## Key Features of Paperless-ngx:

*   **Document Scanning & Import:** Easily scan or import documents from various sources.
*   **Automated Processing:** Automatically extract text, recognize document types, and perform OCR.
*   **Powerful Search:** Quickly find documents with advanced search capabilities.
*   **Organization:** Organize documents with tags, categories, and custom metadata.
*   **Web Interface:** Access your documents from anywhere with a user-friendly web interface.
*   **Open Source:** Benefit from a community-driven project with continuous improvements.

[Explore all features and screenshots in the Documentation](https://docs.paperless-ngx.com/#features)

## Getting Started

The easiest way to deploy Paperless-ngx is using `docker compose`. The configuration files are available in the [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose).

To get started quickly, use our install script:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

Detailed installation guides are available in the [documentation](https://docs.paperless-ngx.com/setup/#installation).

## Migration

Migrating from Paperless-ng is simple.  Refer to the [migration documentation](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx) for more details.

## Documentation

Comprehensive documentation is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

We welcome contributions!  Help us improve Paperless-ngx by:

*   Reporting bugs and suggesting improvements.
*   Contributing code, documentation, or translations.
*   Supporting the project through the [Matrix Room](https://matrix.to/#/#paperless:matrix.org)

### Community Support

Get involved and join the [Matrix Room](https://matrix.to/#/#paperless:matrix.org). If you want to contribute to the project on an ongoing basis there are multiple [teams](https://github.com/orgs/paperless-ngx/people) (frontend, ci/cd, etc) that could use your help so please reach out!

### Translation

Help translate Paperless-ngx into your language on [Crowdin](https://crowdin.com/project/paperless-ngx).

### Feature Requests

Share and vote on feature requests in [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).

### Bugs

Report bugs by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues) or [starting a discussion](https://github.com/paperless-ngx/paperless-ngx/discussions).

## Related Projects

See the [wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects) for a list of related projects.

## Important Security Note

> Paperless-ngx *should not* be run on an untrusted host due to security considerations.  Sensitive information is stored in clear text without encryption. Please run on a local server in your own home with backups.

<p align="right">This project is supported by:<br/>
  <a href="https://m.do.co/c/8d70b916d462" style="padding-top: 4px; display: block;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_white.svg" width="140px">
      <source media="(prefers-color-scheme: light)" srcset="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_blue.svg" width="140px">
      <img src="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_black_.svg" width="140px">
    </picture>
  </a>
</p>