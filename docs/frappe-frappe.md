<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework: Low-Code Web Development for Real-World Applications</h1>
    <p><em>Build powerful web applications quickly with the Frappe Framework, a Python and JavaScript based low-code framework.</em></p>
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
</div>

## About Frappe Framework

Frappe Framework is a full-stack, open-source web application framework built using Python and JavaScript, paired with MariaDB.  It's designed to accelerate web application development by providing a rich set of features and tools.  Initially created for ERPNext, Frappe is a powerful solution for building complex, data-driven applications.

<br>
[View the original repository](https://github.com/frappe/frappe)

## Key Features of Frappe Framework

*   **Full-Stack Development**:  Develop both front-end and back-end components seamlessly within a single framework, streamlining the development process.
*   **Built-in Admin Interface**: Leverage a pre-built, customizable admin dashboard, saving time and effort in managing application data.
*   **Role-Based Permissions**:  Implement robust user and role management to control access and permissions, ensuring data security and compliance.
*   **REST API**: Benefit from automatically generated RESTful APIs for all models, facilitating easy integration with external systems and services.
*   **Customizable Forms and Views**:  Create and modify forms and views flexibly using server-side scripting and client-side JavaScript to match specific requirements.
*   **Report Builder**:  Empower users to generate custom reports without writing code, enhancing data analysis and insights.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for a simple, user-friendly, and sophisticated open-source platform to host Frappe applications.

Frappe Cloud handles installation, upgrades, monitoring, maintenance, and support for your Frappe deployments.  It's a full-featured developer platform that allows you to manage and control multiple Frappe deployments.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self Hosting

### Docker

**Prerequisites:** Docker, Docker Compose, Git.  Refer to [Docker Documentation](https://docs.docker.com) for detailed Docker setup instructions.

**Steps:**

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the Docker Compose command:

    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site should be accessible on `localhost:8080`. Use the default login credentials below:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The recommended way to get started is using the install script for bench, which will install all dependencies (e.g., MariaDB).  See [Frappe Bench](https://github.com/frappe/bench) for more details.

The script will generate new passwords for the Frappe "Administrator" user, the MariaDB root user, and the frappe user. These passwords are displayed and saved to `~/frappe_passwords.txt`.

### Local Setup

Follow these steps to set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, run the following commands:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open the URL `http://frappe.localhost:8000/app` in your browser; the app should be running.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext through courses provided by the maintainers and the community.
2.  [Official Documentation](https://docs.frappe.io/framework) - Extensive documentation covering all aspects of Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe Framework community to ask questions and share knowledge.
4.  [buildwithhussain.com](https://buildwithhussain.com) - See Frappe Framework in action, used to build world-class web applications.

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