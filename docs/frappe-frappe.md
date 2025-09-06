<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>

 **Build powerful, real-world web applications quickly with the Frappe Framework, a low-code, full-stack solution.**
</div>

<div align="center">
	<a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-success.svg"></a>
	<a href="https://codecov.io/gh/frappe/frappe"><img src="https://codecov.io/gh/frappe/frappe/branch/develop/graph/badge.svg?token=XoTa679hIj"/></a>
    <a href="https://github.com/frappe/frappe">
      <img src="https://img.shields.io/github/stars/frappe/frappe?style=social" alt="GitHub stars">
    </a>
</div>
<div align="center">
	<img src=".github/hero-image.png" alt="Hero Image" />
</div>
<div align="center">
    <a href="https://frappe.io/framework">Website</a>
    -
    <a href="https://docs.frappe.io/framework">Documentation</a>
</div>

## Frappe Framework: Low-Code Web Development for Python & JavaScript

Frappe Framework is a powerful, open-source, full-stack web application framework that utilizes Python and MariaDB on the server side with a tightly integrated client-side library. Designed for building robust and scalable applications, Frappe simplifies web development with its low-code approach.  Built as the foundation for ERPNext, Frappe allows developers to build complex applications while focusing on business logic instead of repetitive coding tasks.

**[View the Frappe Framework on GitHub](https://github.com/frappe/frappe)**

### Key Features of Frappe Framework

*   **Full-Stack Framework:** Develop both front-end and back-end components seamlessly within a single framework, streamlining your workflow.
*   **Built-in Admin Interface:** Quickly manage application data with a pre-built, customizable admin dashboard, saving valuable development time.
*   **Role-Based Permissions:** Implement fine-grained access control and user management with a robust role-based permission system.
*   **REST API Generation:**  Automatically generate RESTful APIs for all your models, making integration with other systems effortless.
*   **Customizable Forms and Views:** Tailor your application's forms and views using server-side scripting and client-side JavaScript for a personalized user experience.
*   **Report Builder:** Empower users to create custom reports without writing code using the powerful, intuitive report builder.
*   **Object-Relational Mapping (ORM):**  Frappe uses an ORM that abstracts away complexities of database interactions, allowing developers to work at a higher level of abstraction.
*   **Web Forms and Pages:**  Quickly create and manage web forms and pages, facilitating rapid prototyping and deployment.
*   **Automated Email and Notifications:**  Built-in systems for automated email sending and in-app notifications.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting: Frappe Cloud

For a hassle-free experience, explore [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.

Frappe Cloud handles installation, updates, monitoring, maintenance, and support for your deployments. It offers a fully-featured developer platform with the ability to manage multiple Frappe deployments.

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

**Prerequisites:** docker, docker-compose, git. Refer to the [Docker Documentation](https://docs.docker.com) for Docker setup instructions.

**Steps:**

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run Docker Compose:

    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site should be accessible at `localhost:8080`. Use the following credentials to access the site:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The easiest method involves using the Frappe installation script for bench, which installs all dependencies (e.g., MariaDB). See [Frappe Bench](https://github.com/frappe/bench) for more details.

This script will generate new passwords for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (displayed and saved to `~/frappe_passwords.txt`).

### Local Setup

To set up the repository locally, follow these steps:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```
2.  In a separate terminal window, run:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```
3.  Open the URL `http://frappe.localhost:8000/app` in your browser to view the running application.

## Learning and Community

1.  [Frappe School](https://frappe.school): Learn Frappe Framework and ERPNext through community-led and maintainer courses.
2.  [Official Documentation](https://docs.frappe.io/framework): Comprehensive documentation for Frappe Framework.
3.  [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe Framework user and service provider community.
4.  [buildwithhussain.com](https://buildwithhussain.com): Watch Frappe Framework being used to build real-world web applications.

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