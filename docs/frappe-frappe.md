<div align="center" markdown="1">
    <img src=".github/framework-logo-new.svg" width="80" height="80"/>
    <h1>Frappe Framework: Low-Code Web Development Powerhouse</h1>
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

## Frappe Framework: Build Real-World Web Apps Faster

Frappe Framework is a powerful, open-source, low-code web application framework that empowers developers to build robust and scalable applications using Python and JavaScript. Built on a semantic foundation, Frappe allows for rapid development of complex applications with a focus on user experience and data consistency.  [Explore the Frappe Framework on GitHub](https://github.com/frappe/frappe).

### Key Features of Frappe Framework:

*   **Full-Stack Development**: Develop both front-end and back-end components within a single framework.
*   **Low-Code Approach**: Minimize code writing with built-in features like the Admin Interface, automated APIs, and customizable forms.
*   **Built-in Admin Interface**: Get a ready-to-use, customizable admin dashboard to manage your application data efficiently.
*   **Role-Based Permissions**: Implement granular access control with a comprehensive user and role management system.
*   **REST API Generation**: Automatically generates RESTful APIs for your models, ensuring seamless integration with other services.
*   **Customization Options**: Adapt forms and views to your specific needs using server-side scripting and client-side JavaScript.
*   **Reporting Tools**: Create custom reports easily with the integrated report builder.

<details>
<summary>Screenshots</summary>

![List View](.github/fw-list-view.png)
![Form View](.github/fw-form-view.png)
![Role Permission Manager](.github/fw-rpm.png)
</details>

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for hassle-free hosting of your Frappe applications.  It provides easy installation, upgrades, and maintenance.

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

**Prerequisites:** Docker, Docker Compose, and Git.  For Docker setup details, see the [Docker Documentation](https://docs.docker.com).

**Installation:**

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the application using Docker Compose:

    ```bash
    docker compose -f pwd.yml up -d
    ```

**Access:**  Your site will be accessible on `localhost:8080`.  Use the default login credentials below:

*   **Username:** Administrator
*   **Password:** admin

For ARM-based Docker setups, refer to the [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) documentation.

## Development Setup

### Manual Install

The Easy Way: Use the Frappe install script (bench) to automatically install dependencies, including MariaDB.  See [Frappe Bench Documentation](https://github.com/frappe/bench) for more details.

The script will generate new passwords for the Frappe "Administrator" user, the MariaDB root user, and the frappe user (passwords are saved in `~/frappe_passwords.txt`).

### Local Installation

Follow these steps to set up the repository locally:

1.  Set up bench using the [Installation Steps](https://docs.frappe.io/framework/user/en/installation) and start the server:

    ```bash
    bench start
    ```

2.  In a separate terminal window, run:

    ```bash
    # Create a new site
    bench new-site frappe.localhost
    ```

3.  Access the application in your browser at `http://frappe.localhost:8000/app`.

## Learning and Community

*   [Frappe School](https://frappe.school): Learn Frappe Framework and ERPNext through courses and tutorials.
*   [Official Documentation](https://docs.frappe.io/framework): Comprehensive documentation for Frappe Framework.
*   [Discussion Forum](https://discuss.frappe.io/): Engage with the Frappe Framework community.
*   [buildwithhussain.com](https://buildwithhussain.com):  Explore real-world applications of Frappe Framework.

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
Key improvements and SEO considerations:

*   **Strong Title & Hook:**  The title directly targets a search term ("Frappe Framework") and the hook immediately explains the core benefit (build real-world web apps faster).
*   **Clear Headings:** Uses H2 and H3 headings for better organization and readability, improving SEO.
*   **Keyword Optimization:** Includes relevant keywords like "low-code," "web application framework," "Python," "JavaScript," and "open-source."
*   **Bulleted Key Features:** Makes the core benefits easily scannable for users and search engines.
*   **Concise Language:** Streamlines the text while retaining key information.
*   **Links:**  Keeps the existing links, with some adjusted for clarity.  Includes a link back to the original GitHub repo at the beginning.
*   **Call to Action:** The overview encourages exploration of the framework.
*   **Structure:** Improved organization of the "Production Setup" and "Development Setup" sections.
*   **Alt Text:** Ensures images have descriptive alt text.