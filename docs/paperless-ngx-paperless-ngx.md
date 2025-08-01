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

**Paperless-ngx is a powerful, open-source document management system that lets you organize and access your documents with ease.** Get started with Paperless-ngx on [GitHub](https://github.com/paperless-ngx/paperless-ngx).

Paperless-ngx is the official successor to the original [Paperless](https://github.com/the-paperless-project/paperless) & [Paperless-ng](https://github.com/jonaswinkler/paperless-ng) projects, built to keep your digital documents organized and accessible. Consider joining our community!

Thanks to DigitalOcean, a demo is available at [demo.paperless-ngx.com](https://demo.paperless-ngx.com) using login `demo` / `demo`. _Note: demo content is reset frequently and confidential information should not be uploaded._

## Key Features

*   **Automated Document Processing:** Automatically imports and processes your documents from various sources.
*   **Full-Text Search:** Quickly find documents with powerful search capabilities.
*   **Organized Archiving:** Tag, categorize, and manage your documents efficiently.
*   **OCR (Optical Character Recognition):** Convert scanned documents into searchable text.
*   **Web-Based Access:** Access your documents from anywhere with a web browser.
*   **User-Friendly Interface:** An intuitive interface that makes managing your documents simple.

## Getting Started

The easiest way to deploy Paperless-ngx is with `docker compose`.  See the  [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose) for configuration options.

You can get started quickly with the installation script:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

More details and step-by-step guides for alternative installation methods can be found in [the documentation](https://docs.paperless-ngx.com/setup/#installation).

Migrating from Paperless-ng is easy!  Consult the [migration documentation](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx) for details.

## Documentation

Comprehensive documentation is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

We welcome contributions from the community!  Whether it's bug fixes, enhancements, or visual improvements, your help is appreciated.  See the [documentation](https://docs.paperless-ngx.com/development/) for more information.

*   **Community Support:**  Reach out and get involved in the [Matrix Room](https://matrix.to/#/#paperless:matrix.org) or on GitHub.  Several teams are open to contributors.
*   **Translation:** Help translate Paperless-ngx into your language on [Crowdin](https://crowdin.com/project/paperless-ngx).
*   **Feature Requests:** Submit and vote on feature requests via [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).
*   **Bugs:**  Report bugs by [opening an issue](https://github.com/paperless-ngx/paperless-ngx/issues) or starting a discussion.

## Related Projects

See [the wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects) for a list of related projects and compatible software.

## Important Security Note

>   Document scanners are typically used to scan sensitive documents. **Paperless-ngx should never be run on an untrusted host** because information is stored in clear text without encryption. No guarantees are made regarding security (but we do try!) and you use the app at your own risk.
>
>   **The safest way to run Paperless-ngx is on a local server in your own home with backups in place.**

<p align="right">This project is supported by:<br/>
  <a href="https://m.do.co/c/8d70b916d462" style="padding-top: 4px; display: block;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_white.svg" width="140px">
      <source media="(prefers-color-scheme: light)" srcset="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_blue.svg" width="140px">
      <img src="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_black_.svg" width="140px">
    </picture>
  </a>
</p>