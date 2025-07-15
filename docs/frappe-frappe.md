<div align="center" markdown="1">
  <img src=".github/framework-logo-new.svg" width="80" height="80"/>
  <h1>Frappe Framework</h1>

  **Build powerful, real-world web applications with ease using the Frappe Framework, a low-code solution built on Python and JavaScript.**
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

Frappe Framework is a full-stack, low-code web application framework designed for rapid development of complex, data-driven applications. Built with Python and MariaDB on the server-side, and a tightly integrated client-side library, Frappe provides a robust and flexible environment for building business applications. Originally created to power ERPNext, Frappe's architecture is based on semantic data modeling, ensuring consistency and extensibility for your applications.

[Explore the original repository on GitHub](https://github.com/frappe/frappe)

### Key Features of Frappe Framework:

*   ✅ **Full-Stack Development:** Develop both front-end and back-end components within a single framework.
*   ✅ **Built-in Admin Interface:** Get a pre-built admin dashboard with customizable features, streamlining data management.
*   ✅ **Role-Based Permissions:** Secure your application with a powerful user and role management system for fine-grained access control.
*   ✅ **REST API Generation:** Automatically generates a RESTful API for all models, enabling seamless integration with external systems.
*   ✅ **Customizable Forms and Views:** Easily tailor forms and views using server-side scripting and client-side JavaScript to match your specific needs.
*   ✅ **Report Builder:** Create custom reports without writing code, empowering users to analyze data effectively.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly, open-source platform for hosting Frappe applications. It handles installation, upgrades, monitoring, and support for your deployments.

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

**Prerequisites:** docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

To get started using Docker:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a few minutes, your site should be accessible on localhost port: 8080. Use the default login credentials:
- Username: Administrator
- Password: admin

For ARM-based Docker setups, see [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

## Development Setup

### Manual Install

**Easily set up your development environment with our install script for bench, which installs all dependencies (e.g., MariaDB).** See [https://github.com/frappe/bench](https://github.com/frappe/bench) for details.

The script creates new passwords for the Frappe "Administrator" user, the MariaDB root user, and the frappe user, and saves them to `~/frappe_passwords.txt`.

### Local

Follow these steps to set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, run these commands:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open the URL `http://frappe.localhost:8000/app` in your browser to view the running application.

## Learning and Community

*   📚 [Frappe School](https://frappe.school): Access courses by the framework maintainers or community members.
*   📖 [Official Documentation](https://docs.frappe.io/framework): Find in-depth documentation.
*   💬 [Discussion Forum](https://discuss.frappe.io/): Connect with the Frappe community.
*   📺 [buildwithhussain.com](https://buildwithhussain.com): Watch the framework being used in real-world projects.

## Contributing

*   ✍️ [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   🛡️ [Report Security Vulnerabilities](https://frappe.io/security)
*   🤝 [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   🌐 [Translations](https://crowdin.com/project/frappe)

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