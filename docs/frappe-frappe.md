<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework</h1>
</div>

<div align="center">
    **Build powerful, real-world web applications quickly with Frappe, a low-code framework using Python and JavaScript.**
</div>

<div align="center">
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="MIT License"></a>
    <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Code Coverage"/></a>
</div>
<div align="center">
    <img src=".github/hero-image.png" alt="Frappe Framework Hero Image" />
</div>
<div align="center">
    <a href="https://frappe.io/framework">Website</a> |
    <a href="https://docs.frappe.io/framework">Documentation</a> |
    <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

## About Frappe Framework

Frappe Framework is a full-stack, open-source web application framework that utilizes Python and MariaDB on the server-side, coupled with a tightly integrated client-side library. It's designed for building robust and scalable applications, including ERPNext, the framework's flagship application. Frappe's core philosophy centers around defining the *meaning* of your data, leading to more consistent, extensible, and maintainable applications.

### Key Features

*   **Full-Stack Development**: Develop both front-end and back-end within a single framework, streamlining your development process.
*   **Built-in Admin Interface**: Save time and effort with a pre-built, customizable admin dashboard for efficient data management.
*   **Role-Based Permissions**: Implement granular user and role management to control access and permissions securely.
*   **REST API**: Automatically generate RESTful APIs for seamless integration with other systems and services.
*   **Customizable Forms and Views**: Tailor forms and views with server-side scripting and client-side JavaScript for a personalized user experience.
*   **Report Builder**: Create custom reports effortlessly using a powerful reporting tool, eliminating the need for extensive coding.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for a hassle-free, open-source platform to host your Frappe applications. It handles installation, upgrades, monitoring, maintenance, and support.

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

**Prerequisites**: docker, docker-compose, git. For Docker setup details, refer to the [Docker Documentation](https://docs.docker.com).

**Steps**:

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the docker-compose command:

    ```bash
    docker compose -f pwd.yml up -d
    ```

Your site should be accessible on localhost port: 8080. Use the following default credentials:

*   **Username**: Administrator
*   **Password**: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

Use the bench install script to install all dependencies:

*   See [Frappe Bench](https://github.com/frappe/bench) for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

#### Local Setup

Follow these steps for local setup:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, run:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Access your app at: `http://frappe.localhost:8000/app`

## Learning and Community

*   [Frappe School](https://frappe.school) - Learn Frappe Framework from community and core maintainers.
*   [Official documentation](https://docs.frappe.io/framework) - Extensive documentation.
*   [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com) - Explore real-world applications.

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
Key improvements and SEO considerations:

*   **Clear Headline & Hook:**  The opening sentence immediately conveys what Frappe is and its key benefits.  Includes the target keywords: "Frappe Framework," "web application," "low-code," "Python," and "JavaScript."
*   **Target Keywords:** The text is rich with relevant keywords throughout the document, improving search visibility.
*   **Structured with Headings and Subheadings:**  Organized content is easier to read and helps search engines understand the document's structure.
*   **Bulleted Key Features:** This format makes it easy for potential users to quickly scan and understand the framework's core capabilities.
*   **Links:** Internal and external links provide context and encourage exploration. Includes a clear link to the original repo.
*   **Concise and Actionable:**  The text is written to be engaging and inform the user.
*   **Alt Text:** All images now include descriptive alt text for accessibility and SEO.
*   **Concise "About" Section:** Provides a brief but effective overview of Frappe's purpose and philosophy.
*   **Clear Call to Action (Implicit):**  By explaining the value and how to get started, the README encourages users to try out Frappe.