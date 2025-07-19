<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open Source HR and Payroll Software</h2>
    <p>Manage your entire employee lifecycle with ease using this comprehensive HRMS solution.</p>
</div>

<div align="center">
    <img src=".github/hrms-hero.png" alt="Frappe HR Screenshot"/>
</div>

<div align="center">
    <a href="https://frappe.io/hr">Website</a>
    -
    <a href="https://docs.frappe.io/hr/introduction">Documentation</a>
</div>

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

[View the original repository on GitHub](https://github.com/frappe/hrms)

## Frappe HR: Your Complete Open-Source HRMS Solution

Frappe HR is a modern, open-source HR and Payroll software designed to streamline all aspects of human resource management. From employee onboarding to payroll processing and performance management, Frappe HR offers a comprehensive suite of features to empower your HR team and drive organizational excellence.  Built on the robust Frappe Framework, it's a flexible and scalable solution for businesses of all sizes.

## Key Features

*   **Employee Lifecycle Management:** Simplify employee onboarding, manage promotions, transfers, and offboarding with ease. Streamline the entire employee journey, from hire to retire.
*   **Leave and Attendance Tracking:** Configure custom leave policies, automate attendance tracking, and manage time-off requests efficiently. Ensure accurate attendance records with geolocation features.
*   **Expense Claims and Advances:** Manage employee expense claims and advances, including multi-level approval workflows. Seamlessly integrates with ERPNext accounting for streamlined financial management.
*   **Performance Management:** Track employee goals, align them with key result areas (KRAs), and conduct performance appraisals effectively. Facilitate regular feedback and drive employee growth.
*   **Payroll & Taxation:** Automate payroll processing, configure income tax slabs, and generate accurate salary slips. Manage complex payroll scenarios with ease.
*   **Frappe HR Mobile App:** Empower your employees with the Frappe HR mobile app. Enable on-the-go leave applications and approvals, check-in/check-out functionality, and access to employee profiles.

<details open>
<summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png" alt="Frappe HR Appraisal Screenshot"/>
    <img src=".github/hrms-requisition.png" alt="Frappe HR Requisition Screenshot"/>
    <img src=".github/hrms-attendance.png" alt="Frappe HR Attendance Screenshot"/>
    <img src=".github/hrms-salary.png" alt="Frappe HR Salary Screenshot"/>
    <img src=".github/hrms-pwa.png" alt="Frappe HR PWA Screenshot"/>
</details>

### Built With

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework written in Python and Javascript, providing a solid foundation for building web applications.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library for a modern and intuitive user interface.

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. It handles installation, upgrades, monitoring, and support.

<div>
    <a href="https://frappecloud.com/hrms/signup" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

## Development Setup

### Docker

1.  **Prerequisites:** Docker, docker-compose, and git installed. Refer to [Docker documentation](https://docs.docker.com/).
2.  **Commands:**
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```
3.  Access the application at `http://localhost:8000` using the following credentials:
    *   Username: `Administrator`
    *   Password: `admin`

### Local

1.  **Set up bench:** Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
    ```bash
    bench start
    ```
2.  **In a separate terminal:**
    ```bash
    bench new-site hrms.local
    bench get-app erpnext
    bench get-app hrms
    bench --site hrms.local install-app hrms
    bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn the Frappe Framework and ERPNext.
2.  [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
3.  [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://t.me/frappehr) - Get instant help from the user community.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Logo and Trademark Policy

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