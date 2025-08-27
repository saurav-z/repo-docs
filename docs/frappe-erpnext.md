# ERPNext: Open-Source ERP for Business Management

**Manage your entire business with ERPNext, a powerful, intuitive, and open-source Enterprise Resource Planning (ERP) system.** ([View the original repository](https://github.com/frappe/erpnext))

[![Learn on Frappe School](https://img.shields.io/badge/Frappe%20School-Learn%20ERPNext-blue?style=flat-square)](https://frappe.school)
[![CI](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml/badge.svg?event=schedule)](https://github.com/frappe/erpnext/actions/workflows/server-tests-mariadb.yml)
[![docker pulls](https://img.shields.io/docker/pulls/frappe/erpnext-worker.svg)](https://hub.docker.com/r/frappe/erpnext-worker)

<div align="center">
  <img src="./erpnext/public/images/v16/hero_image.png" alt="ERPNext Hero Image" width="800px"/>
</div>

[Live Demo](https://erpnext-demo.frappe.cloud/api/method/erpnext_demo.erpnext_demo.auth.login_demo) | [Website](https://frappe.io/erpnext) | [Documentation](https://docs.frappe.io/erpnext/)

## Key Features of ERPNext

ERPNext is a comprehensive ERP solution designed to streamline business operations. Here's what you can achieve with it:

*   **Accounting:** Simplify financial management with tools for transactions, reporting, and analysis.
*   **Order Management:** Track inventory, manage sales, orders, and fulfillments.
*   **Manufacturing:** Optimize production cycles, track materials, and handle subcontracting.
*   **Asset Management:** Manage your organization's assets, from IT infrastructure to equipment.
*   **Projects:** Manage projects on time and on budget. Track tasks, timesheets, and issues.

<details open>
<summary>More</summary>
    <img src="https://erpnext.com/files/v16_bom.png" alt="BOM" />
    <img src="https://erpnext.com/files/v16_stock_summary.png" alt="Stock Summary" />
    <img src="https://erpnext.com/files/v16_job_card.png" alt="Job Card" />
    <img src="https://erpnext.com/files/v16_tasks.png" alt="Tasks" />
</details>

## Under the Hood

ERPNext is built on the following technologies:

*   **Frappe Framework:** A full-stack web application framework (Python/JavaScript) providing a robust foundation. ([Frappe Framework](https://github.com/frappe/frappe))
*   **Frappe UI:**  A Vue.js-based UI library for a modern user interface. ([Frappe UI](https://github.com/frappe/frappe-ui))

## Production Setup

### Managed Hosting

Try [Frappe Cloud](https://frappecloud.com) for hassle-free hosting, maintenance, and support.

<div>
    <a href="https://erpnext-demo.frappe.cloud/app/home" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Self-Hosted

#### Docker

Prerequisites: Docker, Docker Compose, and Git.  See the [Docker Documentation](https://docs.docker.com) for setup details.

1.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/frappe_docker
    cd frappe_docker
    ```

2.  Run the Docker Compose command:

    ```bash
    docker compose -f pwd.yml up -d
    ```

After a few minutes, your site will be accessible on `localhost:8080`. Use the following credentials:

*   **Username:** Administrator
*   **Password:** admin

See [Frappe Docker](https://github.com/frappe/frappe_docker?tab=readme-ov-file#to-run-on-arm64-architecture-follow-this-instructions) for ARM based docker setup.

## Development Setup

### Manual Install

Install Bench and dependencies. See [https://github.com/frappe/bench](https://github.com/frappe/bench) for details. New passwords will be created and saved to `~/frappe_passwords.txt`.

### Local

1.  Follow [Installation Steps](https://frappeframework.com/docs/user/en/installation) to setup Bench. Then start the server:

    ```bash
    bench start
    ```

2.  In a new terminal, create a new site:

    ```bash
    bench new-site erpnext.localhost
    ```

3.  Get and install the ERPNext app:

    ```bash
    bench get-app https://github.com/frappe/erpnext
    bench --site erpnext.localhost install-app erpnext
    ```

4.  Access the application at `http://erpnext.localhost:8000/app`.

## Learning and Community

*   [Frappe School](https://school.frappe.io) - Learn ERPNext and Frappe Framework.
*   [Official Documentation](https://docs.erpnext.com/) - Comprehensive ERPNext documentation.
*   [Discussion Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://erpnext_public.t.me) - Get instant support.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)
*   [Translations](https://crowdin.com/project/frappe)

## Logo and Trademark Policy

Review the [Logo and Trademark Policy](TRADEMARK_POLICY.md).

<div align="center" style="padding-top: 0.75rem;">
  <a href="https://frappe.io" target="_blank">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/Frappe-white.png">
      <img src="https://frappe.io/files/Frappe-black.png" alt="Frappe Technologies" height="28"/>
    </picture>
  </a>
</div>
```
Key improvements and explanations:

*   **SEO Optimization:** The title uses the primary keyword "ERPNext" and includes relevant terms like "Open-Source ERP," "Business Management," and "ERP."
*   **One-Sentence Hook:**  The introductory sentence grabs attention and clearly states what ERPNext is.
*   **Clear Headings:**  Uses descriptive headings for easy navigation.
*   **Bulleted Key Features:** Highlights the main functionalities in an easily digestible format.
*   **Concise Language:**  Improved wording for better readability.
*   **Call to Action:**  The links to the demo, website, and documentation are prominently displayed.
*   **Organized Sections:** The layout is logically structured, making it easy for users to find specific information.
*   **Contextual Explanations:** Provides brief explanations for the benefits of each section, making it user-friendly.
*   **Alt text for images:** Improves accessibility and SEO.
*   **Clearer Instructions:** Improves the docker instructions.
*   **Links to the original repo:** added to the beginning of the README.
*   **Added descriptive text to the images for context.**