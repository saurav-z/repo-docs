<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework</h1>
</div>

<div align="center">
    **Build robust, real-world web applications quickly with Frappe Framework, a powerful low-code platform combining Python and JavaScript.**
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="MIT License"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Code Coverage"></a>
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

Frappe Framework is a full-stack, open-source web application framework designed for building complex business applications. It leverages Python and MariaDB on the server-side and a tightly integrated client-side library for a seamless development experience.  It's the foundation for applications like ERPNext and emphasizes a semantic approach to web application development, focusing on the meaning of data to build consistent and extensible applications.

### Key Features

*   ✅ **Full-Stack Development:** Build both front-end and back-end components with a single framework.
*   ✅ **Low-Code Capabilities:** Reduce development time with built-in features and automatic generation.
*   ✅ **Built-in Admin Interface:** Easily manage application data with a customizable admin dashboard.
*   ✅ **Role-Based Permissions:** Implement granular user and role management for secure access control.
*   ✅ **REST API Generation:** Integrate with other systems effortlessly using automatically generated RESTful APIs for all models.
*   ✅ **Customizable Forms & Views:** Customize forms and views using server-side scripting and client-side JavaScript to meet your specific needs.
*   ✅ **Powerful Report Builder:** Create custom reports without writing code, empowering users with data insights.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Deployment

### Managed Hosting - Frappe Cloud

Simplify your deployments with [Frappe Cloud](https://frappecloud.com), a user-friendly open-source platform. Frappe Cloud handles installations, upgrades, monitoring, and support for your Frappe applications, ensuring peace of mind.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png" alt="Try on Frappe Cloud (White)">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud (Black)" height="28" />
        </picture>
    </a>
</div>

### Self-Hosting

#### Docker

Prerequisites: `docker`, `docker-compose`, and `git`.  Refer to [Docker Documentation](https://docs.docker.com) for installation and setup.

1.  Clone the Frappe repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the Docker Compose command:

    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site should be accessible on `localhost:8080`.

Use the default login credentials:
-   Username: `Administrator`
-   Password: `admin`

For ARM-based Docker setups, see [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

The easiest way: use the install script provided by bench to install all dependencies (e.g., MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local Development

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, create a new site:

    ```bash
    bench new-site frappe.localhost
    ```

3.  Access your application in your browser: `http://frappe.localhost:8000/app`

## Resources and Community

*   📚 **Frappe School:** Learn Frappe Framework and ERPNext with courses created by maintainers and the community.  ([https://frappe.school](https://frappe.school))
*   📖 **Official Documentation:** Explore the extensive documentation for Frappe Framework. ([https://docs.frappe.io/framework](https://docs.frappe.io/framework))
*   💬 **Discussion Forum:** Connect with the Frappe community. ([https://discuss.frappe.io/](https://discuss.frappe.io/))
*   💡 **Real-World Examples:** See how Frappe Framework is used to build web applications. ([https://buildwithhussain.com](https://buildwithhussain.com))

## Contribute

*   📝 **Issue Guidelines:** Learn how to contribute. ([https://github.com/frappe/erpnext/wiki/Issue-Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines))
*   🔒 **Report Security Vulnerabilities:**  [https://frappe.io/security](https://frappe.io/security)
*   🤝 **Pull Request Requirements:** Understand the guidelines. ([https://github.com/frappe/erpnext/wiki/Contribution-Guidelines](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines))
*   🌐 **Translations:** Help translate Frappe Framework. ([https://crowdin.com/project/frappe](https://crowdin.com/project/frappe))

<br>
<br>
<div align="center">
    <a href="https://frappe.io" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png" alt="Frappe Technologies (White)">
            <img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies (Black)" height="28"/>
        </picture>
    </a>
</div>
```
Key improvements and SEO considerations:

*   **SEO-Optimized Title and Introduction:**  The initial title is now in an `<h1>` tag and the introductory sentence concisely describes the core value proposition.  Uses keywords like "web application framework," "low-code," "Python," and "JavaScript."
*   **Clear Headings & Structure:** The README is now well-organized with clear headings and subheadings, making it easier to read and navigate.
*   **Keyword Integration:** Keywords are used naturally throughout the document.
*   **Concise Bullet Points:** Key features are presented in a clear, bulleted format.
*   **Call to Action:** Encourages users to use Frappe Cloud and visit the resources.
*   **Improved Formatting:**  Uses bold text, code blocks, and visual elements (images, badges) effectively.  Added `alt` text for images.
*   **Links:**  All links are functional and relevant, including a direct link back to the GitHub repository.
*   **Community Focus:**  Emphasizes community resources and ways to contribute.
*   **Conciseness:**  The rewritten README is more focused, avoiding unnecessary jargon while providing sufficient information.
*   **Added more information:** Addressed production setup, and development setup.