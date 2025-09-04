<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework: Low-Code Web Development in Python & JavaScript</h1>
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
    <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

## Frappe Framework: Build Powerful Web Applications Faster

Frappe Framework is a full-stack, low-code web framework built with Python and JavaScript, designed to streamline web application development and empower developers to build complex, real-world applications with ease.  This framework, inspired by the Semantic Web, emphasizes metadata-driven development for building consistent and extensible applications.

**[Explore the Frappe Framework on GitHub](https://github.com/frappe/frappe)**

### Key Features:

*   **Full-Stack Development:** Develop both front-end and back-end components within a single framework, enhancing efficiency.
*   **Built-in Admin Interface:** Save time with a pre-built, customizable admin dashboard for streamlined data management.
*   **Role-Based Permissions:** Implement robust user and role management to control access and permissions.
*   **REST API Generation:** Benefit from automatically generated RESTful APIs for easy integration with other systems.
*   **Customizable Forms & Views:** Tailor forms and views using server-side scripting and client-side JavaScript.
*   **Powerful Report Builder:** Create custom reports without the need for coding.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting
Frappe Cloud offers a user-friendly platform for hosting Frappe applications, handling installations, upgrades, monitoring, and maintenance.

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
**Prerequisites:** docker, docker-compose, git.

**Steps:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```
After a couple of minutes, access your site on localhost:8080 with these default login credentials:
- Username: Administrator
- Password: admin

## Development Setup

### Manual Install

Follow the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) for bench setup and start the server:
```bash
bench start
```

Then, in a separate terminal window:
```bash
bench new-site frappe.localhost
```
Open the URL `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

1.  [Frappe School](https://frappe.school) - Courses for learning the Frappe Framework.
2.  [Official Documentation](https://docs.frappe.io/framework) - Comprehensive documentation.
3.  [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
4.  [buildwithhussain.com](https://buildwithhussain.com) - Watch real-world Frappe application development.

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