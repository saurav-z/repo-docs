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

# Paperless-ngx: Your Open-Source Document Management System

Paperless-ngx is a powerful, open-source document management system that helps you effortlessly organize and access all your documents.  [Visit the original repo](https://github.com/paperless-ngx/paperless-ngx) for more details.

**Key Features:**

*   **Automated Document Processing:** Automatically import, OCR (Optical Character Recognition), and tag your documents.
*   **Full-Text Search:** Quickly find any document using keywords and phrases.
*   **Organized Archiving:**  Categorize and tag documents for efficient organization and retrieval.
*   **User-Friendly Interface:** Easily manage your documents through an intuitive web interface.
*   **Open-Source and Self-Hosted:**  Gain full control over your data and ensure privacy.
*   **Migration-Friendly:** Seamlessly migrate from the original Paperless and Paperless-ng projects.
*   **Demo Available:** Experience Paperless-ngx firsthand at [demo.paperless-ngx.com](https://demo.paperless-ngx.com) (login: `demo` / `demo`). Note: demo content is reset frequently.

<br>

# Getting Started

The easiest way to deploy Paperless-ngx is using Docker Compose.

*   **Docker Compose Installation:** Use the configuration files in the [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose) to pull the image from the GitHub container registry.
*   **Quick Install Script:**  Get started quickly with our install script:
    ```bash
    bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
    ```
*   **Detailed Installation Guides:**  Find comprehensive instructions for other installation methods in the [documentation](https://docs.paperless-ngx.com/setup/#installation).
*   **Migration:** Easily migrate from Paperless-ng. See the [documentation on migrating](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx) for more details.

<br>

### Documentation

Comprehensive documentation is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

# Contributing

We welcome contributions! Your contributions are vital to Paperless-ngx's success.

*   **Bug Fixes and Enhancements:**  We're always looking for improvements.
*   **Feature Discussions:**  Discuss any major changes beforehand.
*   **Get Involved:** The [documentation](https://docs.paperless-ngx.com/development/) provides info on getting started.

## Community Support

Join our community for support and collaboration:

*   **GitHub:** Engage with the team here.
*   **Matrix Room:** Connect with other users in the [Matrix Room](https://matrix.to/#/#paperless:matrix.org).
*   **Teams:** There are multiple [teams](https://github.com/orgs/paperless-ngx/people) (frontend, ci/cd, etc) that could use your help.

## Translation

Help translate Paperless-ngx into your language:

*   **Crowdin:** Contribute translations through [Crowdin](https://crowdin.com/project/paperless-ngx).

## Feature Requests

Suggest and vote on new features:

*   **GitHub Discussions:** Submit feature requests via [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).

## Bugs

Report and discuss bugs:

*   **GitHub Issues:**  Report bugs by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues).
*   **GitHub Discussions:** Ask questions via [discussions](https://github.com/paperless-ngx/paperless-ngx/discussions).

# Related Projects

Explore compatible projects and software:

*   **Wiki:** Find related projects on the user-maintained [wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects).

# Important Note: Security and Responsible Use

> Document scanners are typically used to scan sensitive documents like your social insurance number, tax records, invoices, etc. **Paperless-ngx should never be run on an untrusted host** because information is stored in clear text without encryption. No guarantees are made regarding security (but we do try!) and you use the app at your own risk.
> **The safest way to run Paperless-ngx is on a local server in your own home with backups in place**.