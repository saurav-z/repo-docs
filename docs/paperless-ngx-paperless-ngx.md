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

**Paperless-ngx is a powerful and user-friendly document management system that helps you digitize, organize, and easily access your documents, all while reducing paper clutter.** (Visit the [original repository](https://github.com/paperless-ngx/paperless-ngx) for more information.)

Paperless-ngx is the official successor to the original [Paperless](https://github.com/the-paperless-project/paperless) & [Paperless-ng](https://github.com/jonaswinkler/paperless-ng) projects, designed to promote a collaborative environment for advancing and supporting the project.

Thanks to the generous folks at [DigitalOcean](https://m.do.co/c/8d70b916d462), a demo is available at [demo.paperless-ngx.com](https://demo.paperless-ngx.com) using login `demo` / `demo`. _Note: demo content is reset frequently and confidential information should not be uploaded._

## Key Features

*   **Document Scanning and Import:** Easily import documents from scanners, emails, and file uploads.
*   **OCR (Optical Character Recognition):** Automatically convert scanned documents into searchable text.
*   **Full-Text Search:** Quickly find documents using keywords, tags, and metadata.
*   **Tagging and Organization:** Categorize and organize documents with tags, custom fields, and flexible filing structures.
*   **User-Friendly Interface:** Enjoy an intuitive and easy-to-navigate web interface.
*   **Open Source and Self-Hosted:** Maintain complete control over your data with self-hosting capabilities.

A full list of [features](https://docs.paperless-ngx.com/#features) and [screenshots](https://docs.paperless-ngx.com/#screenshots) are available in the [documentation](https://docs.paperless-ngx.com/).

## Getting Started

The easiest way to deploy paperless is `docker compose`. The files in the [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose) are configured to pull the image from the GitHub container registry.

If you'd like to jump right in, you can configure a `docker compose` environment with our install script:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

More details and step-by-step guides for alternative installation methods can be found in [the documentation](https://docs.paperless-ngx.com/setup/#installation).

Migrating from Paperless-ng is easy, just drop in the new docker image! See the [documentation on migrating](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx) for more details.

## Documentation

The comprehensive documentation for Paperless-ngx is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

We welcome contributions of all kinds!  Bug fixes, feature enhancements, and visual improvements are always appreciated.

*   **Community Support:**  Join us on [Github](https://github.com/paperless-ngx/paperless-ngx) and in the [Matrix Room](https://matrix.to/#/#paperless:matrix.org) to connect with other users and contributors.  We also have various [teams](https://github.com/orgs/paperless-ngx/people) dedicated to different aspects of the project, so if you're interested in ongoing contribution, please reach out!
*   **Translation:**  Help make Paperless-ngx accessible to everyone by contributing translations on [Crowdin](https://crowdin.com/project/paperless-ngx).
*   **Feature Requests:**  Share your ideas and vote on existing feature requests via [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).
*   **Bugs:**  Report bugs and ask questions by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues) or [starting a discussion](https://github.com/paperless-ngx/paperless-ngx/discussions).

## Related Projects

For a user-maintained list of related projects and software compatible with Paperless-ngx, please see [the wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects).

## Important Security Note

> Document scanners are typically used to scan sensitive documents like your social insurance number, tax records, invoices, etc. **Paperless-ngx should never be run on an untrusted host** because information is stored in clear text without encryption. No guarantees are made regarding security (but we do try!) and you use the app at your own risk.
> **The safest way to run Paperless-ngx is on a local server in your own home with backups in place**.