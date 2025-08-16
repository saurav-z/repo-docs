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

# Paperless-ngx: The Open-Source Document Management System

Paperless-ngx is a powerful document management system that transforms your paper documents into an easily searchable digital archive.  [Visit the original repository](https://github.com/paperless-ngx/paperless-ngx).

**Key Features:**

*   **Automated Document Import:** Automatically import documents from various sources, including scanners and email.
*   **Full-Text Search:** Quickly find documents using keywords and phrases.
*   **Optical Character Recognition (OCR):** Converts scanned documents into searchable text.
*   **Tagging and Organization:** Organize documents with tags, categories, and custom metadata.
*   **User-Friendly Interface:** Easy-to-use web interface for managing your documents.
*   **Open Source & Self-Hosted:**  Take control of your documents with a self-hosted solution.

<img src="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/docs/assets/screenshots/documents-smallcards.png" alt="Paperless-ngx Interface" width="80%">

## Getting Started

The easiest way to deploy Paperless-ngx is using `docker compose`.  The necessary files can be found in the [`/docker/compose` directory](https://github.com/paperless-ngx/paperless-ngx/tree/main/docker/compose).

You can quickly set up a `docker compose` environment using the install script:

```bash
bash -c "$(curl -L https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/main/install-paperless-ngx.sh)"
```

For detailed installation guides, explore the [documentation](https://docs.paperless-ngx.com/setup/#installation). Migrating from Paperless-ng is straightforward; see the [migration documentation](https://docs.paperless-ngx.com/setup/#migrating-to-paperless-ngx).

## Documentation

Comprehensive documentation is available at [https://docs.paperless-ngx.com](https://docs.paperless-ngx.com/).

## Contributing

Contributions are welcome!  Help us improve Paperless-ngx.  

### Community Support

Join the community on Github and the [Matrix Room](https://matrix.to/#/#paperless:matrix.org).

### Translation

Help translate Paperless-ngx into your language on [Crowdin](https://crowdin.com/project/paperless-ngx).

### Feature Requests

Suggest new features via [GitHub Discussions](https://github.com/paperless-ngx/paperless-ngx/discussions/categories/feature-requests).

### Bugs

Report bugs via [GitHub Issues](https://github.com/paperless-ngx/paperless-ngx/issues).

## Related Projects

Explore related projects and compatible software in the [wiki](https://github.com/paperless-ngx/paperless-ngx/wiki/Related-Projects).

## Important Security Note

> Document scanners are often used for sensitive documents.  **Never run Paperless-ngx on an untrusted host.** Information is stored in clear text without encryption, so use it at your own risk. The recommended setup is a local server in your home with backups.
>
>  The safest way to run Paperless-ngx is on a local server in your own home with backups in place.

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

*   **SEO-Optimized Title:**  Uses the primary keyword "Document Management System" and the project name.
*   **One-Sentence Hook:** Immediately grabs attention and describes the core functionality.
*   **Clear Headings:** Uses `##` for clear sectioning and improved readability.
*   **Bulleted Key Features:**  Highlights the essential features in a concise, scannable format.
*   **Visual Enhancement:**  Keeps the logo and adds the screenshot, improving visual appeal and understanding.
*   **Concise Language:**  Avoids unnecessary words and phrases.
*   **Stronger Call to Action:** Encourages users to visit the original repository.
*   **Keyword Density:** Naturally incorporates relevant keywords like "open source," "self-hosted," "OCR," "searchable," and "digital archive".
*   **Security Emphasis:** The security note is now a bit more prominent and strongly worded, which is crucial for the project's success.
*   **Formatting:** Added whitespace for readability.