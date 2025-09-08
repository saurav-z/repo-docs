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

**Frappe Framework** is a robust, low-code web framework built with Python and JavaScript, designed to streamline the development of real-world applications.

### Key Features

*   **Full-Stack Framework:** Develop complete web applications with both front-end and back-end capabilities using a single framework.
*   **Built-in Admin Interface:** Quickly manage your application data with a customizable, pre-built admin dashboard.
*   **Role-Based Permissions:** Secure your application with a comprehensive user and role management system.
*   **REST API:** Automatically generate RESTful APIs for seamless integration with other systems.
*   **Customizable Forms and Views:** Tailor forms and views to your specific needs using server-side scripting and client-side JavaScript.
*   **Report Builder:** Empower users with a powerful reporting tool to create custom reports without coding.
*   **Semantic Web Approach:** Build more consistent and extensible apps based on the underlying system's semantics, not just user interactions.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Experience hassle-free Frappe application hosting with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for open-source deployments.

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

**Prerequisites:** docker, docker-compose, git. Refer to [Docker Documentation](https://docs.docker.com) for setup details.

**Installation:**

```bash
git clone https://github.com/frappe/frappe_docker
cd frappe_docker
docker compose -f pwd.yml up -d
```

After a few minutes, access your site at `localhost:8080`. Use the default credentials:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM-based Docker setup.

## Development Setup

### Manual Install

The easiest way: our install script for bench installs dependencies like MariaDB. See [bench](https://github.com/frappe/bench) for details.

New passwords will be created for Frappe "Administrator", MariaDB root, and the frappe user (the script displays the passwords and saves them to `~/frappe_passwords.txt`).

### Local Setup

Follow these steps to set up the repository locally:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open `http://frappe.localhost:8000/app` in your browser.

## Learning and Community

*   [Frappe School](https://frappe.school) - Courses on Frappe Framework and ERPNext.
*   [Official documentation](https://docs.frappe.io/framework) - Comprehensive framework documentation.
*   [Discussion Forum](https://discuss.frappe.io/) - Connect with the Frappe community.
*   [buildwithhussain.com](https://buildwithhussain.com) - See Frappe Framework in action.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://frappe.io/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

**[Explore the Frappe Framework on GitHub](https://github.com/frappe/frappe)**

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

*   **SEO Optimization:**  Keywords like "web framework," "Python," "JavaScript," "low-code," and relevant features are integrated naturally. The title and headings are also optimized.
*   **Clear Hook:** The opening sentence immediately highlights the value proposition: "Build Powerful Web Applications with Ease."
*   **Organized Structure:**  Uses clear headings and subheadings for readability and easier navigation.
*   **Bulleted Key Features:** Makes the main benefits easy to scan.
*   **Concise Language:**  Removed unnecessary words and streamlined descriptions.
*   **Call to Action:** Includes a direct link back to the original repository.
*   **Improved Formatting:**  Uses consistent formatting (code blocks, bolding, etc.) for better presentation.
*   **Removed Redundancy:** Combined some similar information.
*   **Emphasis on Value:**  Focuses on what the framework *does* (build apps) and *how* it benefits the user (with ease).