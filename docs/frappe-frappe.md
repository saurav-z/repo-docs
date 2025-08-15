<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework: Low-Code Web App Development</h1>
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="MIT License"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Codecov Coverage"></a>
</div>
<div align="center">
    <img src=".github/hero-image.png" alt="Frappe Framework Hero Image" />
</div>
<div align="center">
    <a href="https://frappe.io/framework">Website</a>
    -
    <a href="https://docs.frappe.io/framework">Documentation</a>
    -
    <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

## Build Powerful Web Applications Quickly with Frappe Framework

Frappe Framework is a full-stack, low-code web application framework built on Python and JavaScript, designed for rapid development of real-world applications.  Based on semantic web principles, Frappe helps you build consistent, extensible applications with ease.

**Key Features:**

*   **Full-Stack Development:** Develop both front-end and back-end with a single framework using Python on the server-side and JavaScript on the client-side.
*   **Built-in Admin Interface:** Get a ready-to-use, customizable admin dashboard for efficient data management.
*   **Role-Based Permissions:** Implement comprehensive user and role management for robust access control.
*   **REST API:** Automatically generate RESTful APIs for seamless integration with other services.
*   **Customizable Forms and Views:** Tailor forms and views using server-side scripting and client-side JavaScript.
*   **Report Builder:** Create custom reports without writing code using the powerful reporting tool.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for hassle-free hosting, offering easy setup, upgrades, monitoring, and support.

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

**Prerequisites:** Docker, Docker Compose, Git. Refer to [Docker Documentation](https://docs.docker.com) for details.

**Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Your site should be accessible on `localhost:8080`. Use "Administrator" and "admin" for default login credentials.

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

Use the `bench` install script for easy dependency setup. See [bench installation steps](https://docs.frappe.io/framework/user/en/installation).

New passwords will be created for Frappe "Administrator", MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local

1.  Follow the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) to setup bench and start the server:
    ```bash
    bench start
    ```
2.  In a separate terminal:
    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```
3.  Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

*   [Frappe School](https://frappe.school): Courses on Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.frappe.io/framework): Comprehensive framework documentation.
*   [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com): Learn by watching Frappe in action.

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
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
            <img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
        </picture>
    </a>
</div>
```

Key changes and improvements:

*   **SEO Optimization:** Included keywords like "low-code," "web application framework," "Python," and "JavaScript."
*   **Concise Hook:**  A clear, compelling one-sentence introduction.
*   **Structured Headings:**  Improved readability with clear headings and subheadings.
*   **Bulleted Key Features:**  Easy-to-scan list of core functionalities.
*   **Call to Action:**  Emphasized learning and community resources.
*   **GitHub Link:** Added the link back to the original repository.
*   **Removed Redundancy:** Streamlined text and removed unnecessary phrases.
*   **Alt Text for Images:** Added alt text for all images to improve accessibility and SEO.
*   **Formatting:** Added bolding and emphasis for important aspects.
*   **Markdown Compliance:** Corrected markdown syntax for better rendering.