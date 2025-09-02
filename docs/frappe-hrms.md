<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open Source HRMS for Modern Businesses</h2>
    <p>Manage your entire employee lifecycle with Frappe HR, a modern, open-source HR and payroll solution.</p>
</div>

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)
<a href="https://trendshift.io/repositories/10972" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10972" alt="frappe%2Fhrms | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<div align="center">
    <img src=".github/hrms-hero.png"/>
</div>

<div align="center">
    <a href="https://frappe.io/hr">Website</a>
    -
    <a href="https://docs.frappe.io/hr/introduction">Documentation</a>
    -
    <a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## About Frappe HR

Frappe HR is a comprehensive, open-source Human Resource Management System (HRMS) designed to streamline your HR processes. This user-friendly software offers a wide array of features to manage your workforce, from onboarding to payroll and beyond.  Built on the robust Frappe Framework, it provides a modern and efficient solution for businesses of all sizes.

## Key Features

*   **Employee Lifecycle Management:**  Simplify the employee journey from onboarding to offboarding with features for employee profiles, promotions, transfers, and exit interviews.
*   **Leave and Attendance Tracking:**  Configure flexible leave policies, manage attendance with geolocation, and generate insightful attendance reports.
*   **Expense Claims and Advances:**  Manage employee advances and expense claims with multi-level approval workflows, seamlessly integrated with ERPNext accounting.
*   **Performance Management:** Track goals, align them with key result areas (KRAs), facilitate employee self-evaluations, and simplify the appraisal cycle.
*   **Payroll and Taxation:** Create customizable salary structures, configure income tax slabs, process payroll, handle additional salaries, and generate detailed salary slips.
*   **Mobile Accessibility:** Frappe HR Mobile App allows you to apply for and approve leaves, check-in and check-out, and access employee profiles on the go.

<details open>
  <summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png"/>
    <img src=".github/hrms-requisition.png"/>
    <img src=".github/hrms-attendance.png"/>
    <img src=".github/hrms-salary.png"/>
    <img src=".github/hrms-pwa.png"/>
</details>

## Under the Hood

*   **[Frappe Framework](https://github.com/frappe/frappe):** A powerful full-stack web application framework built with Python and JavaScript, providing a robust foundation for building web applications.
*   **[Frappe UI](https://github.com/frappe/frappe-ui):**  A modern Vue-based UI library, offering a clean and intuitive user interface.

## Production Setup

### Managed Hosting
For a hassle-free deployment, explore [Frappe Cloud](https://frappecloud.com), a user-friendly platform that simplifies hosting, management, and maintenance of Frappe applications.

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
To set up using Docker, ensure you have Docker and docker-compose installed, and then run the following commands:
```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```
Access the HR login screen at `http://localhost:8000` with:
-   Username: `Administrator`
-   Password: `admin`

### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server and keep it running
    ```sh
    $ bench start
    ```
2.  In a separate terminal window, run the following commands
    ```sh
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  You can access the site at `http://hrms.local:8080`

## Learning and Community

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the community of ERPNext users.
*   [Telegram Group](https://t.me/frappehr) - Get instant help from the community.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Logo and Trademark Policy

Please read our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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