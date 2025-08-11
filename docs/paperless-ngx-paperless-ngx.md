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

# Paperless-ngx: The Open-Source Document Management System (DMS)

**Tired of paper clutter? Paperless-ngx is a powerful, open-source document management system that lets you scan, organize, and search your documents with ease.** Find out more on the original repo: [https://github.com/paperless-ngx/paperless-ngx](https://github.com/paperless-ngx/paperless-ngx)

## Key Features

*   **Document Scanning & Import:** Easily scan or upload documents from various sources.
*   **Automated OCR:** Optical Character Recognition (OCR) automatically converts scanned documents into searchable text.
*   **Advanced Tagging & Organization:**  Categorize, tag, and organize documents for efficient retrieval.
*   **Full-Text Search:** Quickly find documents using keywords, dates, or other metadata.
*   **Web-Based Interface:** Access your documents from anywhere with a web browser.
*   **Open Source & Self-Hosted:**  Take control of your data with a self-hosted solution.
*   **User-Friendly Interface:** Provides intuitive and user-friendly experience.
*   **Available in Multiple Languages:** Built with translations in mind.

## Getting Started

Paperless-ngx is simple to install using Docker Compose.  See the following steps for installation:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

Comprehensive setup instructions and alternative installation methods are available in the [official documentation](https://docs.paperless-ngx.com/setup/#installation).

Migrating from Paperless-ng is easy!  Consult the [migration documentation](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx) for details.

### Documentation

The complete Paperless-ngx documentation is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

We welcome contributions! Whether you're interested in bug fixes, new features, or documentation improvements, your help is appreciated.

*   **Community Support:** Get involved in the [Matrix Room](https://matrix.to/#/#paperless:matrix.org) or on GitHub.  Explore the [available teams](https://github.com/orgs/paperless-ngx/people) for ongoing contributions.
*   **Translation:** Help translate Paperless-ngx into your language via [Crowdin](https://crowdin.com/project/paperless-ngx).
*   **Feature Requests:** Submit and discuss new features in [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).
*   **Bugs:** Report bugs by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues) or starting a discussion.

## Related Projects

Explore a list of related projects and compatible software on the [wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects).

## Important Security Note

>  Because of the sensitivity of the documents you scan, **Paperless-ngx should never be run on an untrusted host**. As information is stored in clear text without encryption, there are no security guarantees.
>  **For the best security, run Paperless-ngx on a local server in your home with backups.**