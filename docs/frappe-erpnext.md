<!-- ERPNext - Open-Source ERP Software -->
<div align="center">
    <a href="https://frappe.io/erpnext">
	    <img src="./erpnext/public/images/v16/erpnext.svg" alt="ERPNext Logo" height="80px" width="80px"/>
    </a>
    <h2>ERPNext</h2>
    <p align="center">
        <strong>Unlock your business's potential with ERPNext, a powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system.</strong>
    </p>
    <p align="center">
        <a href="https://frappe.school">
            <img src="https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square" alt="Learn ERPNext on Frappe School">
        </a>
        <br><br>
        <a href="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml">
            <img src="https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule" alt="CI Status">
        </a>
        <a href="https://hub.docker.com/r/frappe/erpnext-worker">
            <img src="https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg" alt="Docker Pulls">
        </a>
    </p>
</div>

<div align="center">
	<img src="./erpnext/public/images/v16/hero_image.png" alt="ERPNext Hero Image"/>
</div>

<div align="center">
	<a href="https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo" target="_blank">Live Demo</a>
	-
	<a href="https://frappe.io/erpnext" target="_blank">Website</a>
	-
	<a href="https://docs.frappe.io/erpnext/" target="_blank">Documentation</a>
    -
    <a href="https://github.com/frappe/erpnext" target="_blank">View on GitHub</a>
</div>

## What is ERPNext?

ERPNext is a 100% open-source ERP system designed to streamline your business operations and boost efficiency. It provides a comprehensive suite of tools to manage various aspects of your business, all in one place.

### Key Features

*   **Accounting:** Comprehensive financial management tools, from transaction recording to financial reporting and analysis.
*   **Order Management:** Track inventory, manage sales orders, handle customer and supplier relationships, and oversee shipments and fulfillment.
*   **Manufacturing:** Simplify production cycles, manage material consumption, implement capacity planning, and handle subcontracting.
*   **Asset Management:** Manage assets throughout their lifecycle, from purchase to disposal, covering infrastructure and equipment.
*   **Projects:** Manage both internal and external projects, tracking tasks, timesheets, issues, and profitability.

<details open>
    <summary>More</summary>
        <img src="https://erpnext.com/files/v16_bom.png" alt="Bill of Materials">
        <img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary">
        <img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card">
        <img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks">
</details>

### Technology Stack

*   **Frappe Framework:** A full-stack web application framework (Python & JavaScript) providing a robust foundation with database abstraction, user authentication, and a REST API. [View on GitHub](https://github.com/frappe/frappe)
*   **Frappe UI:** A modern Vue-based UI library for building user-friendly applications. [View on GitHub](https://github.com/frappe/frappe-ui)

## Getting Started

### Production Setup

#### Managed Hosting (Recommended)

Frappe Cloud offers a simple, user-friendly, and sophisticated platform for hosting Frappe applications. It handles installation, upgrades, monitoring, and maintenance, allowing you to focus on your business.

<div>
    <a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

#### Self-Hosted

##### Docker

**Prerequisites:** Docker, Docker Compose, Git. Refer to [Docker Documentation](https://docs.docker.com) for detailed setup instructions.

**Steps:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```
2.  Run the Docker Compose command:
    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site will be accessible on `localhost:8080`. Use the following default credentials:

*   **Username:** Administrator
*   **Password:** admin

For ARM-based Docker setups, refer to the [Frappe Docker documentation](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions).

### Development Setup

#### Manual Install

Install bench using the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server using `bench start`.

1.  Create a new site:
    ```bash
    bench new-site erpnext.localhost
    ```
2.  Get and install the ERPNext app:
    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```
3.  Access the app: Open `http://erpnext.localhost:8000/app` in your browser.

## Learning and Community

*   **Frappe School:** Learn Frappe Framework and ERPNext through community and maintainer-led courses. ([https://school.frappe.io](https://school.frappe.io))
*   **Official Documentation:** Extensive documentation for ERPNext. ([https://docs.erpnext.com/](https://docs.erpnext.com/))
*   **Discussion Forum:** Engage with the ERPNext community. ([https://discuss.erpnext.com/](https://discuss.erpnext.com/))
*   **Telegram Group:** Get instant help from the ERPNext community. ([https://erpnext_public.t.me](https://erpnext_public.t.me))

## Contribute

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Trademark Policy

Review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

<br />
<br />
<div align="center" style="padding-top: 0.75rem;">
    <a href="https://frappe.io" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
            <img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
        </picture>
    </a>
</div>
```
Key improvements and SEO considerations:

*   **Concise, benefit-driven hook:**  "Unlock your business's potential with ERPNext..." immediately tells the user what they will gain.
*   **Clear headings and subheadings:** Structure makes the information easily scannable.
*   **Keyword Optimization:** Includes terms like "open-source ERP," "ERP system," and key features within the content.
*   **Bulleted Lists:** Improve readability and highlight key features.
*   **Targeted Links:** Links to the demo, website, documentation and back to the original repo, plus relevant community resources, providing a strong SEO profile.
*   **Alt text for images:**  Ensures accessibility and aids search engine understanding of images.
*   **Clear Call to Action:** Prompts users to try the demo and explore resources.
*   **Removed Redundancy:** Streamlined the text to be more concise.
*   **Use of `target="_blank"` on external links:** Opens links in a new tab, keeping users engaged with the main page.
*   **Added "What is ERPNext?" section for clearer introduction.**