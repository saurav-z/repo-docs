<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework</h1>
</div>

<div align="center">
    **Build powerful, real-world web applications quickly with Frappe Framework, a low-code Python and JavaScript framework.**
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="License: MIT"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Code Coverage"></a>
</div>

<div align="center">
    <img src=".github/hero-image.png" alt="Hero Image" />
</div>

<div align="center">
    <a href="https://frappe.io/framework">Website</a>
    -
    <a href="https://docs.frappe.io/framework">Documentation</a>
</div>

## Frappe Framework: The Low-Code Web Framework for Rapid Application Development

Frappe Framework is a full-stack web application framework built with Python and MariaDB on the server-side and a tightly integrated client-side library.  It's designed for building complex, data-driven applications with speed and efficiency. Learn more at the [original repository](https://github.com/frappe/frappe).

### Key Features

*   **Full-Stack Framework:** Develop both front-end and back-end with a unified framework.
*   **Built-in Admin Interface:**  Get a customizable admin dashboard to manage your application data effortlessly.
*   **Role-Based Permissions:** Implement robust user and role management for secure access control.
*   **REST API:** Automatically generate RESTful APIs for seamless integration with other services.
*   **Customizable Forms and Views:** Tailor forms and views with server-side scripting and client-side JavaScript.
*   **Report Builder:**  Create custom reports easily with a powerful, no-code reporting tool.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Experience hassle-free deployments with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.  It handles installation, upgrades, monitoring, and support.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self Hosting

#### Docker

**Prerequisites:**  Docker, Docker Compose, and Git. Refer to the [Docker Documentation](https://docs.docker.com) for installation instructions.

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose command:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Your site should be accessible on `localhost:8080` after a few minutes.  Use the following credentials to log in:

*   **Username:** Administrator
*   **Password:** admin

For ARM-based Docker setups, see [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

Use the install script for bench, which will install all dependencies (e.g., MariaDB). See [bench documentation](https://github.com/frappe/bench) for more details.

The script creates new passwords for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (passwords are displayed and saved to `~/frappe_passwords.txt`).

### Local Development

Follow these steps to set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  In a separate terminal, run:
    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```
3.  Open `http://frappe.localhost:8000/app` in your browser to view the running application.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
2.  [Official documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - See Frappe in action.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://frappe.io/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
4.  [Translations](https://crowdin.com/project/frappe)

<br>
<br>
<div align="center">
    <a href="https://frappe.io" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
            <img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
        </picture>
    </a>
</div>
```
Key improvements and SEO considerations:

*   **Clear, Concise Title:** Added "The Low-Code Web Framework for Rapid Application Development" to the title for better SEO.
*   **One-Sentence Hook:** Included a compelling opening sentence to grab the reader's attention.
*   **Keywords:** Incorporated relevant keywords like "low-code," "Python," "JavaScript," "web application framework," and "rapid application development."
*   **Bulleted Key Features:** Used a bulleted list to highlight the most important features, making them easy to scan.
*   **Structured Headings:** Organized the content with clear headings and subheadings for readability and SEO.
*   **Concise Descriptions:**  Rewrote descriptions to be more concise and focused.
*   **Alt Text:** Added alt text to images for accessibility and SEO.
*   **Stronger Calls to Action:**  Implied calls to action by providing links for "Learn More," "Documentation," "Website," and "Try Frappe Cloud".
*   **Link Back to Original Repo:** Added a clear reference to the original repository.