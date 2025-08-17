# Paperless-ngx: Your Digital Document Management Solution

Paperless-ngx is a powerful open-source document management system (DMS) that helps you organize, search, and archive your documents digitally.  [View the original repository on GitHub](https://github.com/paperless-ngx/paperless-ngx).

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

*   **Automated Document Processing:** Automatically imports, OCRs (Optical Character Recognition), and indexes your documents.
*   **Full-Text Search:**  Quickly find any document using keyword searches within the content.
*   **Organized Filing:** Easily tag, categorize, and archive documents for efficient organization.
*   **Web-Based Interface:**  Access your documents from any device with a web browser.
*   **Open Source & Self-Hosted:**  Maintain full control over your data and privacy.
*   **User-Friendly Interface:** Streamlines the process of document management.

### Demo

Experience Paperless-ngx firsthand at [demo.paperless-ngx.com](https://demo.paperless-ngx.com) using the login `demo` / `demo`.  *Please note that demo content is reset frequently.*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards.png">
  <img src="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards.png">
</picture>


## Getting Started

The easiest way to deploy paperless is using `docker compose`. Instructions and files can be found in the [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose).

Alternatively, you can use the install script:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

Detailed installation guides are available in the [documentation](https://docs.paperless-ngx.com/setup/#installation).

## Documentation

Comprehensive documentation for Paperless-ngx is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

Contributions are welcome!  Help improve Paperless-ngx through bug fixes, enhancements, and more.

*   **Community Support:** Connect with other users and contributors in the [Matrix Room](https://matrix.to/#/#paperless:matrix.org).
*   **Translations:** Help translate Paperless-ngx into your language on [Crowdin](https://crowdin.com/project/paperless-ngx).
*   **Feature Requests:** Submit and discuss feature requests in [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).
*   **Bugs:** Report bugs by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues).

## Related Projects

Explore compatible software and related projects on the [wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects).

## Important Security Note

> **Warning:**  Paperless-ngx is designed for use on trusted hosts and should not be run on an untrusted server, as it stores data without encryption.  Backups are crucial, and running Paperless-ngx locally is the safest approach.