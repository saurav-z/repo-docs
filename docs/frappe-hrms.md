<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open Source HRMS & Payroll Software</h2>
    <p>
        <b>Manage your entire employee lifecycle with Frappe HR, a modern, open-source HR and payroll solution.</b>
    </p>
</div>

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

<a href="https://trendshift.io/repositories/10972" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10972" alt="frappe%2Fhrms | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<div align="center">
    <img src=".github/hrms-hero.png" alt="Frappe HR Hero Image"/>
</div>

<div align="center">
    <a href="https://frappe.io/hr">Website</a>
    -
    <a href="https://docs.frappe.io/hr/introduction">Documentation</a>
    -
    <a href="https://github.com/frappe/hrms"><b>View on GitHub</b></a>
</div>

## About Frappe HR

Frappe HR is a comprehensive, open-source Human Resource Management System (HRMS) designed to streamline and automate your HR processes.  With over 13 modules, it provides everything you need to manage your employees, from onboarding to payroll.

## Key Features

*   **Employee Lifecycle Management:**  Efficiently onboard employees, manage promotions, transfers, and conduct exit interviews.
*   **Leave and Attendance Tracking:** Configure custom leave policies, manage attendance with geolocation, and track leave balances.
*   **Expense Claims and Advances:** Manage employee advances, claim expenses, and configure multi-level approval workflows.
*   **Performance Management:** Track goals, align with key result areas (KRAs), and simplify appraisal cycles.
*   **Payroll & Taxation:** Create salary structures, configure tax slabs, run payroll, and generate salary slips.
*   **Mobile App:**  Manage leaves, check-in/check-out, and access employee profiles on the go with the Frappe HR mobile app.

<details open>
    <summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png" alt="Appraisal Screenshot"/>
    <img src=".github/hrms-requisition.png" alt="Requisition Screenshot"/>
    <img src=".github/hrms-attendance.png" alt="Attendance Screenshot"/>
    <img src=".github/hrms-salary.png" alt="Salary Screenshot"/>
    <img src=".github/hrms-pwa.png" alt="PWA Screenshot"/>
</details>

## Technical Underpinnings

Frappe HR is built upon robust open-source technologies:

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework providing the foundation for Frappe HR.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui):  A modern Vue-based UI library offering a user-friendly interface.

## Getting Started

### Production Setup

For easy hosting, try [Frappe Cloud](https://frappecloud.com), a managed platform for Frappe applications.  It handles installation, upgrades, and maintenance.

<div>
    <a href="https://frappecloud.com/hrms/signup" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Development Setup

Choose your preferred setup method:

#### Docker

1.  Install [Docker](https://docs.docker.com/) and [Docker Compose](https://docs.docker.com/compose/).
2.  Clone the repository: `git clone https://github.com/frappe/hrms`
3.  Navigate to the Docker directory: `cd hrms/docker`
4.  Run: `docker-compose up`
5.  Access the application at `http://localhost:8000` with the credentials:  Username: `Administrator`, Password: `admin`.

#### Local Setup

1.  Install Frappe Framework by following the [installation steps](https://frappeframework.com/docs/user/en/installation).
2.  Start the bench server: `bench start`
3.  In a separate terminal:
    ```bash
    bench new-site hrms.local
    bench get-app erpnext
    bench get-app hrms
    bench --site hrms.local install-app hrms
    bench --site hrms.local add-to-hosts
    ```
4.  Access the site at `http://hrms.local:8080`

## Resources and Community

*   [Frappe School](https://frappe.school) - Learn the Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help and connect with other users.

## Contributing

We welcome contributions!  Please review the following:

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Trademark Policy

Please review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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

*   **Clear Hook:** Starts with a compelling one-sentence introduction that highlights the core benefit.
*   **Keywords:** Uses relevant keywords like "open source HRMS," "payroll software," and key features throughout the summary.
*   **Headings:**  Uses clear, descriptive headings (e.g., "About Frappe HR," "Key Features," "Technical Underpinnings") to structure the content and improve readability.
*   **Bulleted Lists:**  Uses bulleted lists to highlight key features and benefits, making them easy to scan.
*   **Concise Language:**  Uses concise language to convey information effectively.
*   **Alt Text:** Includes descriptive `alt` text for all images for accessibility and SEO.
*   **Internal Links:** Links back to the original GitHub repository.
*   **External Links:** Includes relevant links to documentation, community resources, and Frappe Cloud for increased engagement and SEO.
*   **Call to Action (Implied):** The clear presentation and feature descriptions encourage users to explore the software.
*   **Removed Unnecessary Content:** Streamlined the content for better readability.
*   **Emphasis on Open Source:** Prominently highlights the "open-source" nature of the software.