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

# Paperless-ngx: The Ultimate Document Management System

**Paperless-ngx helps you effortlessly digitize, organize, and retrieve your documents, freeing you from the clutter of paper.**  Check out the [Paperless-ngx GitHub Repository](https://github.com/paperless-ngx/paperless-ngx) for more information.

Paperless-ngx is the official successor to the original [Paperless](https://github.com/the-paperless-project/paperless) & [Paperless-ng](https://github.com/jonaswinkler/paperless-ng) projects, designed to distribute the responsibility of advancing and supporting the project among a dedicated team. Consider joining us!

Thanks to generous support from [DigitalOcean](https://m.do.co/c/8d70b916d462), you can experience a live demo at [demo.paperless-ngx.com](https://demo.paperless-ngx.com) using credentials `demo` / `demo`.  _Note: Demo content is reset frequently, and it is recommended to not upload confidential information._

## Key Features

*   **Automated Document Scanning & Import:** Easily scan and import documents from various sources.
*   **OCR & Full-Text Search:**  Optical Character Recognition (OCR) converts scanned documents into searchable text, allowing for fast and efficient retrieval.
*   **Organized Tagging & Categorization:**  Organize documents with tags, categories, and custom metadata fields for easy filtering and retrieval.
*   **Secure Storage & Backup:**  Store documents safely with options for backups.
*   **Web-Based Interface:** Access your documents from any device with a web browser.
*   **Open Source & Self-Hosted:** Take control of your documents with a fully open-source and self-hosted solution.

## Getting Started

The simplest way to deploy Paperless-ngx is using `docker compose`. Configuration files are available in the [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose) and are configured to pull the image from the GitHub container registry.

Quickly get started by configuring a `docker compose` environment with our installation script:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

Detailed instructions and other installation methods can be found in [the documentation](https://docs.paperless-ngx.com/setup/#installation).

Migrating from Paperless-ng is straightforward; just use the new Docker image!  See the [migration documentation](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx) for more details.

## Documentation

Comprehensive documentation for Paperless-ngx is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

Contributions are welcome!  Help with bug fixes, enhancements, and visual improvements is highly appreciated. If you plan to implement a major feature, please start a discussion first.  See the [documentation](https://docs.paperless-ngx.com/development/) for information on how to contribute.

### Community Support

The community actively supports Paperless-ngx development. Engage with us on GitHub and in the [Matrix Room](https://matrix.to/#/#paperless:matrix.org).  If you wish to become a regular contributor, explore various teams such as frontend and CI/CD, all of which welcome assistance!

### Translation

Paperless-ngx is available in numerous languages, which are coordinated through Crowdin. Contribute to translations at https://crowdin.com/project/paperless-ngx. For more information, please consult the [CONTRIBUTING.md](https://github.com/paperless-ngx/paperless-ngx/blob/main/CONTRIBUTING.md#translating-paperless-ngx).

### Feature Requests

Submit feature requests via [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests), where you can search existing ideas, propose your own, and vote for features you want.

### Bugs

Report bugs by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues) or starting a discussion if you have questions.

## Related Projects

Find a user-maintained list of related projects and compatible software on the [wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects).

## Important Note Regarding Security

> Document scanners frequently handle sensitive information such as social security numbers, tax records, and invoices.  **Paperless-ngx should only be run on a trusted host** because the data is stored without encryption in plain text. Security is not guaranteed (though we make an effort!), and using the application is at your own risk.
> **The safest approach is to run Paperless-ngx on a local server at home with proper backups.**