<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
    <h1>Frappe Framework: Low-Code Web Development Revolutionized</h1>
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
</div>

## Frappe Framework: Build Powerful Web Applications with Ease

Frappe Framework is a powerful, low-code web application framework, built with Python and JavaScript, perfect for creating real-world, data-driven applications.  Designed to be developer-friendly, Frappe streamlines web application development, allowing you to focus on innovation.  [Learn more on the original repository](https://github.com/frappe/frappe).

**Key Features:**

*   **Full-Stack Development:** Develop both front-end and back-end with a single framework, accelerating your development process.
*   **Built-in Admin Interface:** Enjoy a ready-to-use, customizable admin dashboard, saving time and effort on data management.
*   **Role-Based Permissions:** Implement robust user and role management, ensuring secure and controlled access to application features.
*   **REST API Generation:** Automatically generate RESTful APIs for all your models, simplifying integrations.
*   **Customization Capabilities:** Tailor forms and views using server-side scripting and client-side JavaScript.
*   **Report Builder:** Empower users to create insightful reports without requiring any coding.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for a hassle-free hosting solution.  It handles installations, upgrades, and maintenance.

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

**Prerequisites:** Docker, Docker Compose, Git. Refer to [Docker Documentation](https://docs.docker.com) for setup.

**Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Your site should be accessible on `localhost:8080` after a few minutes.

*   **Default login:**
    *   Username: `Administrator`
    *   Password: `admin`

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

Our install script (bench) simplifies the setup process and installs required dependencies like MariaDB. See [bench documentation](https://github.com/frappe/bench) for more details.

**Note:** Passwords for the Administrator user, MariaDB root user, and the frappe user will be generated and saved in `~/frappe_passwords.txt`.

### Local

To set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server.
    ```bash
    bench start
    ```

2.  In a separate terminal window, run:
    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open `http://frappe.localhost:8000/app` in your browser to see the running app.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext through courses.
2.  [Official documentation](https://docs.frappe.io/framework) - Access comprehensive Frappe Framework documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe Framework community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - See real-world Frappe Framework applications.

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