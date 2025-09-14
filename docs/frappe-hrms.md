<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open Source HR & Payroll Software</h2>
    <p><b>Revolutionize your HR management with Frappe HR, the open-source solution that empowers you to manage your entire employee lifecycle, payroll, and more with ease.</b></p>
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
    <a href="https://github.com/frappe/hrms">GitHub Repository</a>
</div>

## Frappe HR: Your Complete HRMS Solution

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline and optimize your HR processes. Featuring over 13 integrated modules, Frappe HR provides everything you need to manage your employees from onboarding to offboarding.

## Key Features

*   **Employee Lifecycle Management:** Simplify employee onboarding, manage promotions and transfers, and document performance with exit interviews.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, manage regional holidays, enable geo-location check-in/out, and generate comprehensive attendance reports.
*   **Expense Claims and Advances:** Manage employee advances, expense claims, and configure multi-level approval workflows, fully integrated with ERPNext accounting.
*   **Performance Management:** Track goals, align them with key result areas (KRAs), enable self-evaluations, and simplify appraisal cycles.
*   **Payroll & Taxation:** Create salary structures, configure tax slabs, process payroll, handle additional salaries, generate salary slips, and much more.
*   **Mobile Accessibility:** Utilize the Frappe HR Mobile App for on-the-go leave applications, approvals, check-ins/outs, and employee profile access.

<details open>
    <summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png" alt="Appraisal Screenshot"/>
    <img src=".github/hrms-requisition.png" alt="Requisition Screenshot"/>
    <img src=".github/hrms-attendance.png" alt="Attendance Screenshot"/>
    <img src=".github/hrms-salary.png" alt="Salary Screenshot"/>
    <img src=".github/hrms-pwa.png" alt="PWA Screenshot"/>
</details>

## Under the Hood

*   **Frappe Framework:** A robust, full-stack web application framework built with Python and Javascript. ([Frappe Framework GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A modern and responsive Vue-based UI library. ([Frappe UI GitHub](https://github.com/frappe/frappe-ui))

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform to host Frappe applications. It handles installation, upgrades, monitoring, and maintenance.

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

Requires Docker, docker-compose, and git.

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```
2.  Access the application at `http://localhost:8000` with the following credentials:
    *   Username: `Administrator`
    *   Password: `admin`

### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server.
    ```bash
    $ bench start
    ```
2.  In a separate terminal window:
    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn the Frappe Framework and ERPNext.
2.  [Documentation](https://docs.frappe.io/hr) - Comprehensive documentation for Frappe HR.
3.  [User Forum](https://discuss.erpnext.com/) - Engage with the community of ERPNext users and service providers.
4.  [Telegram Group](https://t.me/frappehr) - Get instant help from the community of users.

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