<div align="center" markdown="1">
  <img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
  <h1>Frappe Framework: Low-Code Web Development for Real-World Applications</h1>
</div>

<div align="center">
  <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="MIT License"></a>
  <a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Code Coverage"></a>
</div>
<div align="center">
  <img src=".github/hero-image.png" alt="Frappe Framework Hero Image" />
</div>
<div align="center">
  <a href="https://frappe.io/framework">Website</a> | <a href="https://docs.frappe.io/framework">Documentation</a> | <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

## Frappe Framework: Build Powerful Web Apps with Ease

**Frappe Framework** is a powerful, open-source, full-stack web application framework that empowers developers to build complex, real-world applications using Python and JavaScript. This low-code platform is designed for rapid development and easy maintenance.

### Key Features:

*   **Full-Stack Development:** Develop both the front-end and back-end of your application within a single framework, streamlining the development process.
*   **Built-in Admin Interface:** Quickly create and customize an admin dashboard to manage your application's data, saving valuable development time.
*   **Role-Based Permissions:** Implement robust user and role management to control access and permissions, ensuring data security.
*   **REST API:** Automatically generate a RESTful API for seamless integration with other systems and services.
*   **Customizable Forms and Views:** Design flexible forms and views using server-side scripting and client-side JavaScript to create tailored user experiences.
*   **Report Builder:** Empower users to create custom reports without writing code, providing valuable insights into your data.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting: Frappe Cloud

For a simplified hosting experience, consider [Frappe Cloud](https://frappecloud.com). This platform offers easy installation, upgrades, monitoring, and support for your Frappe applications.

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

**Prerequisites:** docker, docker-compose, git.  See [Docker Documentation](https://docs.docker.com) for setup.

**Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Your site should be accessible on localhost:8080 after a few minutes.  Use these credentials to log in:
*   **Username:** Administrator
*   **Password:** admin

For ARM-based Docker setups, see [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

Use our install script for bench, which handles all dependencies (like MariaDB):  See [bench Documentation](https://github.com/frappe/bench).

The script creates new passwords for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (passwords are displayed and saved to `~/frappe_passwords.txt`).

### Local

1.  Set up bench using the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

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

1.  [Frappe School](https://frappe.school):  Learn Frappe Framework and ERPNext through courses from the maintainers and community.
2.  [Official Documentation](https://docs.frappe.io/framework): Comprehensive documentation for Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe Framework community.
4.  [buildwithhussain.com](https://buildwithhussain.com): See Frappe Framework in action building real-world web apps.

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