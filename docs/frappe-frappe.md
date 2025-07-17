<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework</h1>

    **Build powerful, real-world web applications quickly and efficiently with Frappe, a low-code framework built on Python and JavaScript.**
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="MIT License"/></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Code Coverage"/></a>
</div>
<div align="center">
    <img src=".github/hero-image.png" alt="Frappe Framework Hero Image" />
</div>
<div align="center">
    <a href="https://frappe.io/framework">Website</a>
    -
    <a href="https://docs.frappe.io/framework">Documentation</a>
    -
    <a href="https://github.com/frappe/frappe">GitHub Repository</a>
</div>

## What is Frappe Framework?

Frappe Framework is a full-stack, open-source web application framework that utilizes Python and MariaDB on the server-side and a tightly integrated client-side library. It's designed for building robust and scalable applications, and is the foundation of ERPNext.  Inspired by semantic web principles, Frappe focuses on defining the *meaning* of data alongside its presentation, leading to more consistent, extensible, and maintainable applications.

### Key Features

*   **Full-Stack Development:** Develop both front-end and back-end components within a unified framework.
*   **Low-Code/No-Code Approach**: Frappe allows building applications with minimal coding, increasing speed of development.
*   **Built-in Admin Interface:** Simplify application management with a customizable, pre-built admin dashboard.
*   **Role-Based Permissions:** Implement granular access control with a comprehensive user and role management system.
*   **REST API:** Easily integrate with other systems using automatically generated RESTful APIs for all models.
*   **Customizable Forms and Views:** Tailor forms and views to your specific needs with server-side scripting and client-side JavaScript.
*   **Report Builder:** Create custom reports without writing code using the powerful reporting tool.
*   **Open Source**: Frappe is fully open-source, so you can access and contribute to the source code.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting - Frappe Cloud

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. Frappe Cloud handles installation, upgrades, monitoring, and maintenance, enabling you to focus on development.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png" alt="Try Frappe Cloud (Dark Mode)">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try Frappe Cloud (Light Mode)" height="28" />
        </picture>
    </a>
</div>

### Self Hosting

#### Docker

**Prerequisites**: Ensure you have Docker, Docker Compose, and Git installed. Refer to the [Docker Documentation](https://docs.docker.com) for detailed setup instructions.

To run Frappe using Docker:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a few minutes, your Frappe site will be accessible on `localhost:8080`.  Use the following default credentials:

*   Username: `Administrator`
*   Password: `admin`

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setups.

## Development Setup

### Manual Install

The easiest way to install is via our install script for bench, which handles dependency installation, including MariaDB. See [Frappe Bench](https://github.com/frappe/bench) for details.

The script will create new passwords for the Frappe "Administrator" user, the MariaDB root user, and the frappe user, saving them to `~/frappe_passwords.txt`.

#### Local Development Steps

1.  Set up bench following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, run:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open `http://frappe.localhost:8000/app` in your browser to view your running application.

## Learning and Community

*   [Frappe School](https://frappe.school): Learn Frappe Framework and ERPNext through courses.
*   [Official Documentation](https://docs.frappe.io/framework): Comprehensive documentation.
*   [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com): See Frappe Framework in action.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://frappe.io/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

<br>
<br>
<div align="center">
    <a href="https://frappe.io" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png" alt="Frappe Technologies (Dark Mode)">
            <img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies (Light Mode)" height="28"/>
        </picture>
    </a>
</div>
```
Key improvements and explanations:

*   **SEO Optimization:**  Included keywords like "web application framework," "low-code," "Python," "JavaScript," and "open-source" in the introduction and key feature descriptions.  This will help with search engine rankings.
*   **Concise Hook:**  The one-sentence hook is at the top, immediately grabbing the reader's attention.
*   **Clear Headings & Structure:** Uses clear, descriptive headings and subheadings to improve readability and make it easy to find information.
*   **Bulleted Key Features:**  Provides a quick overview of the framework's capabilities.
*   **Concise Explanations:**  Simplified and clarified the explanations for each feature.
*   **Call to Action:** Includes links to the website, documentation, and GitHub repo to encourage engagement.
*   **Complete Docker Instructions:**  The Docker section now includes a summary of commands, making it easier for users to get started.
*   **Community & Learning Section:** Highlights resources for learning and contributing.
*   **Alt Text for Images:** Added `alt` text to all images for accessibility and SEO.
*   **Clearer Language:** Improved phrasing for better clarity.
*   **Duplicated Information Removal:** Removed some duplicate information.
*   **Dark Mode Support:** Added alt text for dark mode images.
*   **Consistent Formatting:**  Maintained consistent markdown formatting throughout.
*   **GitHub Repo Link:** Added a link to the GitHub repo.