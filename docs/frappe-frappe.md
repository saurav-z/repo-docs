<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework: Low-Code Web Development Powerhouse</h1>
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj"/></a>
</div>
<div align="center">
    <img src=".github/hero-image.png" alt="Hero Image" />
</div>
<div align="center">
    <a href="https://frappe.io/framework">Website</a>
    -
    <a href="https://docs.frappe.io/framework">Documentation</a>
    -
    <a href="https://github.com/frappe/frappe">GitHub Repository</a>
</div>

## Frappe Framework: Build Real-World Web Applications with Ease

Frappe Framework is a **full-stack, low-code web application framework** built with Python and JavaScript, offering a powerful and efficient way to develop robust applications. Designed for rapid development and scalability, Frappe empowers developers to build complex solutions with less code.

### Key Features:

*   **Full-Stack Development:** Develop both front-end and back-end components within a single framework, streamlining the development process.
*   **Built-in Admin Interface:** Save time with a ready-made, customizable admin dashboard for managing application data and settings.
*   **Role-Based Permissions:** Implement granular access control with a robust user and role management system.
*   **REST API Generation:** Automatically generate RESTful APIs for all your models, ensuring seamless integration with other systems.
*   **Customizable Forms and Views:** Tailor your application's forms and views using server-side scripting and client-side JavaScript to match your exact needs.
*   **Report Builder:** Create powerful custom reports quickly, even without extensive coding.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started with Frappe

### Production Setup

#### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com), a hassle-free, open-source platform designed for hosting Frappe applications. It simplifies installations, upgrades, monitoring, and maintenance.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

#### Self Hosting

#### Docker

**Prerequisites:** docker, docker-compose, git. Refer to [Docker Documentation](https://docs.docker.com) for setup.

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the docker-compose command:
    ```bash
    docker compose -f pwd.yml up -d
    ```

Your site should be accessible on `localhost:8080` after a few minutes.  Use the default credentials below.

*   **Username:** Administrator
*   **Password:** admin

For ARM-based Docker setups, refer to [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

### Development Setup

#### Manual Install

The easiest way is to use the install script for bench, which installs all dependencies (e.g. MariaDB).  See [Frappe Bench](https://github.com/frappe/bench) for details.

*   The script creates new passwords for the Frappe "Administrator" user, the MariaDB root user, and the frappe user.  These passwords are saved to `~/frappe_passwords.txt`.

#### Local

To set up the repository locally:

1.  Follow the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) to setup bench and start the server:
    ```bash
    bench start
    ```

2.  In a separate terminal window:
    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community Resources

*   [Frappe School](https://frappe.school) - Courses on Frappe Framework and ERPNext.
*   [Official Documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
*   [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com) - See Frappe in action.

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
Key improvements and explanations:

*   **SEO-Optimized Title:**  Included the keyword "web application framework" and "low-code" in the title for better search visibility.
*   **Concise Hook:** The opening sentence immediately highlights the core value proposition.
*   **Clear Headings:** Uses H2 and H3 tags for improved readability and SEO (makes it easy for search engines to understand the structure).
*   **Bulleted Key Features:** Makes the key selling points easy to scan.
*   **Concise, Actionable Instructions:**  Simplified setup instructions with clearer steps and commands.
*   **Community & Learning:**  Expanded on learning resources to make it clear how new users can learn the framework.
*   **Consistent Formatting:** Applied consistent formatting (e.g., code blocks, bolding) throughout for better readability.
*   **GitHub Link:** Added a link to the original repository for easy navigation.
*   **Removed Unnecessary Elements**: Removed elements that did not contribute to a concise description.
*   **Expanded Description**:  Added more introductory information to give a better overview of what the framework is, what it does, and why users should care.
*   **Emphasis on Benefits**: Framed the features in terms of their benefits to developers (e.g., "Save time," "Seamless integration").

This improved version is much more user-friendly and SEO-friendly, making it easier for potential users to understand the value of Frappe Framework and get started.