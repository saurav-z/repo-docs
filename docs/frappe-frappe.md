<div align="center" markdown="1">
	<img src=".github/framework-logo-new.svg" width="80" height="80"/>
	<h1>Frappe Framework</h1>

 <b>Build powerful, real-world web applications quickly with Frappe Framework, a low-code, full-stack framework.</b>
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

## Frappe Framework: The Low-Code Web Framework for Rapid Application Development

Frappe Framework is a full-stack web application framework built with Python and JavaScript, designed for building robust and scalable applications. Originally developed for ERPNext, Frappe provides a powerful, yet accessible, platform for developers to create real-world solutions.

### Key Features

*   **Full-Stack Development:** Develop both front-end and back-end with a single framework.
*   **Built-in Admin Interface:** Quickly manage application data with a customizable admin dashboard.
*   **Role-Based Permissions:** Implement granular user and role management for secure access control.
*   **REST API Generation:** Automatically create RESTful APIs for seamless integration.
*   **Customizable Forms and Views:** Tailor forms and views using server-side scripting and client-side JavaScript.
*   **Report Builder:** Empower users to create custom reports without writing any code.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting (Frappe Cloud)

For the easiest deployment, try [Frappe Cloud](https://frappecloud.com), a managed hosting platform that handles installation, upgrades, and maintenance.

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

To run Frappe with Docker:

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

Your site should be accessible at `localhost:8080` after a few minutes. Use the following credentials:

*   Username: `Administrator`
*   Password: `admin`

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

Install bench using the installation script found at [Frappe Bench](https://github.com/frappe/bench), which installs all dependencies (e.g. MariaDB).

New passwords will be created for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local Development

1.  Install and start bench:
    ```bash
    bench start
    ```

2.  Create a new site:
    ```bash
    bench new-site frappe.localhost
    ```

3.  Access your app: `http://frappe.localhost:8000/app`

## Learning and Community

*   [Frappe School](https://frappe.school) - Courses on Frappe Framework and ERPNext.
*   [Official documentation](https://docs.frappe.io/framework)
*   [Discussion Forum](https://discuss.frappe.io/)
*   [buildwithhussain.com](https://buildwithhussain.com) - Learn by example.

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

*   **SEO-Optimized Title & Introduction:**  The title now includes "Low-Code" and "Web Framework" to improve searchability. The introduction clearly states the purpose and core benefit.
*   **Concise Summary:** The text is streamlined, focusing on key features and benefits.
*   **Bulleted Key Features:** The "Key Features" section is now a bulleted list for readability and easy scanning.  Keywords are used to aid SEO.
*   **Clear Headings & Structure:**  Uses clear, semantic headings (H2, H3) to organize content, improving readability and SEO.
*   **Call to Action:**  The "Production Setup" section gives deployment options and links to resources.
*   **GitHub Link:**  Added a link back to the GitHub repository.
*   **Removed Redundancy:** Removed duplicated information.
*   **Community & Learning Section:** Included links to important community resources.
*   **Contributing Section:** Clear links to contribution guidelines.
*   **Concise Docker Instructions:**  Simplified the docker setup instructions.
*   **Keywords:** Integrated relevant keywords naturally throughout the text (e.g., "low-code", "full-stack", "web application framework", "Python", "JavaScript").
*   **Formatting:** Improved markdown formatting for better visual appeal.