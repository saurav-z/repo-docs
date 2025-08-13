<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open Source HR & Payroll Software</h2>
    <p>
        <b>Manage your entire employee lifecycle with Frappe HR, the open-source HRMS built for modern businesses.</b>
    </p>

    [![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
    [![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

    <a href="https://trendshift.io/repositories/10972" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10972" alt="frappe%2Fhrms | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<div align="center">
    <img src=".github/hrms-hero.png" alt="Frappe HR Dashboard"/>
</div>

<div align="center">
    <a href="https://frappe.io/hr">Website</a>
    -
    <a href="https://docs.frappe.io/hr/introduction">Documentation</a>
    -
    <a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## Key Features of Frappe HR

Frappe HR is a comprehensive Human Resource Management System (HRMS) packed with features to streamline your HR processes.  It offers over 13 modules designed to cover all aspects of HR from employee management and leave tracking to payroll and performance management.

*   **Employee Lifecycle Management:** Simplify onboarding, manage promotions and transfers, and conduct exit interviews for a complete employee lifecycle.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, integrate regional holidays, use geolocation-based check-in/out, and generate detailed attendance reports.
*   **Expense Claims and Advances:** Manage employee advances, process expense claims, and configure multi-level approval workflows, seamlessly integrated with ERPNext accounting.
*   **Performance Management:** Set and track goals, align goals with key result areas (KRAs), enable employee self-evaluation, and streamline appraisal cycles.
*   **Payroll & Taxation:** Create custom salary structures, configure tax slabs, run payroll, handle additional payments, and view comprehensive salary slips.
*   **Mobile App:** Access key HR functions on the go with the Frappe HR mobile app, including leave applications, approvals, and employee profile access.

<details open>
    <summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png" alt="Frappe HR Appraisal Screen"/>
    <img src=".github/hrms-requisition.png" alt="Frappe HR Requisition Screen"/>
    <img src=".github/hrms-attendance.png" alt="Frappe HR Attendance Screen"/>
    <img src=".github/hrms-salary.png" alt="Frappe HR Salary Screen"/>
    <img src=".github/hrms-pwa.png" alt="Frappe HR PWA Screen"/>
</details>

## Technology Stack

Frappe HR is built on robust and open-source technologies:

*   **Frappe Framework:**  A full-stack web application framework in Python and JavaScript, providing a solid foundation for building web applications, including database management, user authentication, and a REST API. ([Frappe Framework GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A Vue.js-based UI library offering a modern user interface. ([Frappe UI GitHub](https://github.com/frappe/frappe-ui))

## Getting Started

### Managed Hosting

Simplify your setup with [Frappe Cloud](https://frappecloud.com), a user-friendly platform that handles installation, upgrades, and maintenance, offering peace of mind for your Frappe HR deployments.

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

Requires Docker, docker-compose, and Git.  Follow these steps:

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```
2.  Access Frappe HR in your browser at `http://localhost:8000`

    Use these credentials to log in:
    *   Username: `Administrator`
    *   Password: `admin`

#### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
    ```bash
    $ bench start
    ```
2.  In a separate terminal:
    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Resources and Community

*   [Frappe School](https://frappe.school) - Learn the Frappe Framework and ERPNext from community-led courses.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive documentation for Frappe HR.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get quick help from the community.

## Contribute

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Logo and Trademark Policy

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