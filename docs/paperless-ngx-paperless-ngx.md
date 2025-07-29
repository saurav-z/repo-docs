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

# Paperless-ngx: Your Open-Source Document Management Solution

Paperless-ngx is a powerful document management system that helps you **organize, search, and archive your documents digitally, reducing clutter and saving you time.**  [Visit the project's GitHub repository for more details.](https://github.com/paperless-ngx/paperless-ngx)

Paperless-ngx is the official successor to the original [Paperless](https://github.com/the-paperless-project/paperless) & [Paperless-ng](https://github.com/jonaswinkler/paperless-ng) projects and is designed to distribute the responsibility of advancing and supporting the project among a team of people. [Consider joining us!](#community-support)

Thanks to the generous folks at [DigitalOcean](https://m.do.co/c/8d70b916d462), a demo is available at [demo.paperless-ngx.com](https://demo.paperless-ngx.com) using login `demo` / `demo`. _Note: demo content is reset frequently and confidential information should not be uploaded._

## Key Features:

*   **Automated Document Organization:** Automatically import, OCR, and tag your documents.
*   **Full-Text Search:** Quickly find documents using keywords and phrases.
*   **Web-Based Interface:** Access your documents from anywhere with a web browser.
*   **Optical Character Recognition (OCR):** Makes scanned documents searchable.
*   **Tagging and Categorization:** Organize documents with custom tags and categories.
*   **User-Friendly Interface:** An intuitive interface for easy document management.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards.png">
  <img src="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards.png">
</picture>

For a comprehensive list of features and screenshots, please see the official [documentation](https://docs.paperless-ngx.com/).

## Getting Started

The easiest way to deploy Paperless-ngx is using `docker compose`. Configuration files are provided in the [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose).

Quickstart with `docker compose`:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

More detailed installation instructions and alternative methods can be found in the [documentation](https://docs.paperless-ngx.com/setup/#installation).

Migrating from Paperless-ng is straightforward; simply use the new Docker image – detailed migration steps are available in the [documentation](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx).

### Documentation

Comprehensive documentation is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

We welcome contributions!  Whether it's bug fixes, enhancements, or documentation improvements, your help is appreciated.

*   **Community Support:**  Connect with the community on GitHub and the [Matrix Room](https://matrix.to/#/#paperless:matrix.org).
*   **Translation:** Help translate Paperless-ngx into your language via [Crowdin](https://crowdin.com/project/paperless-ngx).
*   **Feature Requests:** Suggest and discuss new features in [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).
*   **Bugs:** Report bugs by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues) or [starting a discussion](https://github.com/paperless-ngx/paperless-ngx/discussions).

## Related Projects

Check out the [wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects) for a list of related projects and compatible software.

## Important Security Note

> **Important:** Paperless-ngx should be run on a trusted host. Because the system stores information in clear text without encryption, **it is not secure to run Paperless-ngx on an untrusted host.** No guarantees are made regarding security, and you use the app at your own risk. We recommend running Paperless-ngx on a local server with backups in place.

<p align="right">This project is supported by:<br/>
  <a href="https://m.do.co/c/8d70b916d462" style="padding-top: 4px; display: block;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_white.svg" width="140px">
      <source media="(prefers-color-scheme: light)" srcset="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_blue.svg" width="140px">
      <img src="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_black_.svg" width="140px">
    </picture>
  </a>
</p>
```
Key improvements and explanations:

*   **SEO Optimization:**  Uses relevant keywords like "document management," "open source," "archive," "searchable," and "organization" throughout the content and in headings.
*   **Clear Hook:** The first sentence is a concise and compelling introduction to the software's purpose.
*   **Headings and Structure:**  Organized with clear headings and subheadings to improve readability and scannability.
*   **Bulleted Key Features:**  Provides a quick overview of the main features.
*   **Concise Language:**  Uses clear and direct language.
*   **Call to Action:** Encourages users to contribute.
*   **Link Back to Repo:** The introduction includes a clear link back to the main GitHub repository.
*   **Enhanced Security Warning:** The security note is emphasized to highlight the importance of the security considerations.
*   **Demo Note Placement:** The demo content is placed on the top, for a new user's ease of access.
*   **Image Enhancement:** The image descriptions were enhanced.