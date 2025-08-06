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

**Paperless-ngx is an open-source document management system that helps you effortlessly organize, archive, and search your documents, freeing you from paper clutter.** ([See the original repo](https://github.com/paperless-ngx/paperless-ngx))

Paperless-ngx is the official successor to the original [Paperless](https://github.com/the-paperless-project/paperless) & [Paperless-ng](https://github.com/jonaswinkler/paperless-ng) projects, and offers a powerful and user-friendly way to manage all your documents digitally.  Join the community and start managing your documents today!

Thanks to the generous folks at [DigitalOcean](https://m.do.co/c/8d70b916d462), a demo is available at [demo.paperless-ngx.com](https://demo.paperless-ngx.com) using login `demo` / `demo`. _Note: demo content is reset frequently and confidential information should not be uploaded._

## Key Features

*   **Automated Document Processing:** Automatically import, OCR (Optical Character Recognition), and tag your documents.
*   **Powerful Search:** Easily search for documents using keywords, tags, document types, and more.
*   **Organized Archiving:** Categorize and organize documents with tags, document types, and custom fields.
*   **User-Friendly Interface:** Intuitive web interface for easy document management.
*   **Open Source and Self-Hosted:** Control your data and privacy by hosting Paperless-ngx on your own server.
*   **Easy Installation:** Simple setup using Docker Compose for quick deployment.
*   **Community Driven:** Benefit from a large and active community supporting and developing the project.

## Getting Started

The easiest way to deploy Paperless-ngx is using `docker compose`.  Files are available in the [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose) to pull the image from the GitHub container registry.

For a quick start, use this install script:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

Detailed installation guides are available in [the documentation](https://docs.paperless-ngx.com/setup/#installation).

Migrating from Paperless-ng is easy; simply deploy the new Docker image! See the [documentation on migrating](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx) for more details.

### Documentation

Comprehensive documentation is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

We welcome contributions! Bug fixes, enhancements, and visual improvements are always appreciated. If you're planning a large feature, please start a discussion.  For more details, see [the documentation on contributing](https://docs.paperless-ngx.com/development/).

### Community Support

Get involved in the community on GitHub and in the [Matrix Room](https://matrix.to/#/#paperless:matrix.org).  To contribute to the project long-term, explore the various [teams](https://github.com/orgs/paperless-ngx/people) and offer your help!

### Translation

Help translate Paperless-ngx into your language on Crowdin: https://crowdin.com/project/paperless-ngx.  See [CONTRIBUTING.md](https://github.com/paperless-ngx/paperless-ngx/blob/main/CONTRIBUTING.md#translating-paperless-ngx) for more details.

### Feature Requests

Submit feature requests via [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests), where you can also search for existing ideas, vote, and contribute your own.

### Bugs

Report bugs by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues) or [starting a discussion](https://github.com/paperless-ngx/paperless-ngx/discussions).

## Related Projects

See [the wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects) for a user-maintained list of related projects and software.

## Important Note

> Document scanners are typically used to scan sensitive documents like your social insurance number, tax records, invoices, etc. **Paperless-ngx should never be run on an untrusted host** because information is stored in clear text without encryption. No guarantees are made regarding security (but we do try!) and you use the app at your own risk.
> **The safest way to run Paperless-ngx is on a local server in your own home with backups in place**.