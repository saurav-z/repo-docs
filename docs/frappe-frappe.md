<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework</h1>
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

## Frappe Framework: Build Powerful Web Applications with Ease

Frappe Framework is a full-stack, low-code web framework that empowers developers to build robust and scalable web applications using Python and JavaScript, designed for real-world application development.  [Visit the original repository](https://github.com/frappe/frappe).

### Key Features

*   **Full-Stack Development**: Develop both front-end and back-end components within a unified framework, streamlining the development process.
*   **Low-Code Approach**:  Minimize the need for extensive coding with built-in features like a customizable admin interface and auto-generated APIs.
*   **Built-in Admin Interface**:  Manage and customize your application data efficiently with a pre-built admin dashboard.
*   **Role-Based Permissions**: Implement granular access control with a comprehensive user and role management system.
*   **REST API**: Benefit from automatically generated RESTful APIs for all models, facilitating easy integration with external services.
*   **Customizable Forms and Views**: Tailor forms and views to your specific needs using server-side scripting and client-side JavaScript.
*   **Report Builder**: Create custom reports without writing code using the powerful built-in reporting tool.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for a hassle-free hosting experience. It's a user-friendly platform that handles installation, upgrades, monitoring, and maintenance of your Frappe applications, providing a fully featured developer platform.

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
Prerequisites: docker, docker-compose, git. Refer [Docker Documentation](https://docs.docker.com) for more details on Docker setup.

Run following commands:

```
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a couple of minutes, site should be accessible on your localhost port: 8080. Use below default login credentials to access the site.
- Username: Administrator
- Password: admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The Easy Way: our install script for bench will install all dependencies (e.g. MariaDB). See https://github.com/frappe/bench for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to ~/frappe_passwords.txt).

### Local

To setup the repository locally follow the steps mentioned below:

1. Setup bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server
   ```
   bench start
   ```

2. In a separate terminal window, run the following commands:
   ```
   # Create a new site
   bench new-site frappe.localhost
   ```

3. Open the URL `http://frappe.localhost:8000/app` in your browser, you should see the app running

## Learning and Community

*   [Frappe School](https://frappe.school) - Access courses to learn Frappe Framework and ERPNext from the maintainers and community.
*   [Official Documentation](https://docs.frappe.io/framework) - Consult the extensive documentation for detailed information.
*   [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe Framework community.
*   [buildwithhussain.com](https://buildwithhussain.com) - Learn through live examples of Frappe Framework being used in the wild.

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