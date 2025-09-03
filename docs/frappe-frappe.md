<!-- Improved & Summarized README for Frappe Framework -->

<div align="center">
	<img src=".github/framework-logo-new.svg" width="80" height="80" alt="Frappe Framework Logo"/>
	<h1>Frappe Framework</h1>

 **Build powerful, real-world web applications with ease using this low-code, full-stack framework built on Python and JavaScript.**
</div>

<div align="center">
	<a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg" alt="License: MIT"></a>
	<a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj" alt="Code Coverage"></a>
	<a href="https://github.com/frappe/frappe"><img src="https://img.shields.io/github/stars/frappe/frappe?style=social" alt="GitHub Stars"></a>
</div>
<div align="center">
	<img src=".github/hero-image.png" alt="Hero Image" />
</div>
<div align="center">
    <a href="https://frappe.io/framework">Website</a>
    -
    <a href="https://docs.frappe.io/framework">Documentation</a>
</div>

## Frappe Framework: The Low-Code Web Development Powerhouse

Frappe Framework is a full-stack, open-source web application framework designed to accelerate development. It uses Python and MariaDB on the server side and a tightly integrated client-side library. Originally built for ERPNext, Frappe offers a robust and flexible platform for building a wide variety of applications.

[**Explore the original repository on GitHub**](https://github.com/frappe/frappe)

## Key Features

*   **Full-Stack Development:** Develop both front-end and back-end logic within a single framework using Python and JavaScript.
*   **Built-in Admin Interface:** Save time with a pre-built, customizable admin dashboard for easy data management.
*   **Role-Based Permissions:** Fine-grained control over user access and permissions to ensure data security.
*   **REST API:** Automatically generates RESTful APIs for all models, enabling seamless integration with other systems.
*   **Customizable Forms and Views:** Tailor forms and views to your specific needs using server-side scripting and client-side JavaScript.
*   **Report Builder:** Create custom reports with an intuitive tool, eliminating the need for extensive coding.
*   **Rapid Development:** Frappe's low-code approach significantly reduces development time and effort.
*   **Open Source:** Benefit from the collaborative power of an open-source community and transparent code.
*   **Scalable Architecture:** Built to handle complex applications and large datasets.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting (Frappe Cloud)

For a hassle-free deployment, consider [Frappe Cloud](https://frappecloud.com). This platform offers easy installation, updates, monitoring, and support for your Frappe applications.

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

**Prerequisites:** Docker, Docker Compose, Git.  Refer to [Docker Documentation](https://docs.docker.com) for setup instructions.

**Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Your site should be accessible on `http://localhost:8080` after a couple of minutes.

**Default Login:**

*   Username: `Administrator`
*   Password: `admin`

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

The Easy Way: Use the install script via `bench`.  See https://github.com/frappe/bench for more details.

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local

Follow these steps to set up the repository locally:

1.  Set up `bench` by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server.

    ```bash
    bench start
    ```

2.  In a separate terminal window, run:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open the URL `http://frappe.localhost:8000/app` in your browser. You should see the app running.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn the framework and ERPNext.
2.  [Official documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - See Frappe Framework in action.

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