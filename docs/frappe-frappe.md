<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework: Low-Code Web Development</h1>
    <p><b>Build robust and scalable web applications with ease using Python and JavaScript.</b></p>
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
	- <a href="https://github.com/frappe/frappe">View on GitHub</a>
</div>

## About Frappe Framework

Frappe Framework is a full-stack, low-code web application framework that empowers developers to build real-world applications quickly and efficiently. Built on Python and MariaDB for the server-side and a tightly integrated client-side library, it provides a comprehensive solution for rapid web application development. Inspired by the Semantic Web, Frappe focuses on the meaning and structure of data, leading to more consistent and extensible applications.

### Key Features

*   ✅ **Full-Stack Development:**  Develop both front-end and back-end components within a single framework, streamlining your development process.
*   ✅ **Built-in Admin Interface:**  Save time with a pre-built, customizable admin dashboard for efficient data management.
*   ✅ **Role-Based Permissions:**  Implement granular access control with a robust user and role management system.
*   ✅ **REST API Generation:**  Automatically generate RESTful APIs for all your models, facilitating seamless integration with other services.
*   ✅ **Customizable Forms & Views:** Tailor forms and views using server-side scripting and client-side JavaScript for a personalized user experience.
*   ✅ **Report Builder:**  Empower users to create custom reports effortlessly with the built-in reporting tool.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Getting Started

### Production Setup

#### Managed Hosting: Frappe Cloud

For a hassle-free deployment, consider [Frappe Cloud](https://frappecloud.com). This platform provides a user-friendly, open-source solution for hosting Frappe applications, handling installations, updates, monitoring, maintenance, and support.

<div>
    <a href="https://frappecloud.com/" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

#### Self-Hosting

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

To set up the repository locally, follow these steps:

1.  Set up bench by following the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, run the following commands:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Open the URL `http://frappe.localhost:8000/app` in your browser; you should see the app running.

## Resources

*   📚 [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext from the various courses by the maintainers or from the community.
*   📖 [Official Documentation](https://docs.frappe.io/framework) - Extensive documentation for Frappe Framework.
*   💬 [Discussion Forum](https://discuss.frappe.io/) - Engage with the Frappe community.
*   📺 [buildwithhussain.com](https://buildwithhussain.com) - Watch Frappe Framework in action, building real-world web apps.

## Contributing

Help improve Frappe!  Here's how:

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
Key improvements and optimizations:

*   **SEO-Friendly Heading and Title:**  Includes the keyword "Frappe Framework" prominently and uses an H1 tag.
*   **Concise Hook:** A one-sentence summary capturing the essence of the framework.
*   **Bulleted Key Features:**  Uses bullet points for readability and scannability, enhancing SEO.  Added emojis for extra visual appeal.
*   **Clear Structure:**  Uses headings and subheadings to organize information, making it easy for users and search engines to understand.
*   **Keyword Optimization:**  Naturally incorporates relevant keywords like "low-code," "web application framework," "Python," and "JavaScript."
*   **Call to Action:**  Includes a clear call to action - Try Frappe Cloud and a link back to original repo
*   **Community Engagement:**  Highlights resources and encourages contributions.
*   **Conciseness:**  Removes unnecessary wording and streamlines information.
*   **Enhanced Formatting:** Improved Markdown for better readability (bolding, details).
*   **Clear Sections:** Clearly separates setup, learning, and contributing sections.