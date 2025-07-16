<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
</div>

## Frappe HR: Open Source HR and Payroll Software for Modern Businesses

Frappe HR is a complete, open-source Human Resources Management System (HRMS) designed to streamline your HR processes and empower your team.  [Explore Frappe HR on GitHub](https://github.com/frappe/hrms)

<div align="center">
    <img src=".github/hrms-hero.png" alt="Frappe HR Screenshot">
</div>

<div align="center">
    <a href="https://frappe.io/hr">Website</a> |
    <a href="https://docs.frappe.io/hr/introduction">Documentation</a>
</div>

## Key Features of Frappe HR

Frappe HR provides a comprehensive suite of modules to manage your entire employee lifecycle.  Here are some of the core capabilities:

*   **Employee Lifecycle Management:** Simplify onboarding, promotions, transfers, and exit interviews, providing a seamless experience for your employees.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, manage attendance with geolocation, and track leave balances and attendance reports.
*   **Expense Claims and Advances:**  Manage employee advances, streamline expense claims with multi-level approval workflows, and integrate seamlessly with ERPNext accounting.
*   **Performance Management:** Track employee goals, align them with key result areas (KRAs), facilitate self-evaluations, and simplify appraisal cycles.
*   **Payroll and Taxation:**  Create salary structures, configure income tax slabs, run payroll, handle additional payments, and provide detailed salary slips.
*   **Frappe HR Mobile App:** Manage HR tasks on the go, including leave applications, approvals, and attendance tracking.

<details open>
    <summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png" alt="Appraisal Screenshot" />
    <img src=".github/hrms-requisition.png" alt="Requisition Screenshot" />
    <img src=".github/hrms-attendance.png" alt="Attendance Screenshot" />
    <img src=".github/hrms-salary.png" alt="Salary Screenshot" />
    <img src=".github/hrms-pwa.png" alt="PWA Screenshot" />
</details>

## Under the Hood

Frappe HR is built on robust open-source technologies:

*   **[Frappe Framework](https://github.com/frappe/frappe):** A full-stack web application framework written in Python and JavaScript, providing a solid foundation for building and deploying web applications.
*   **[Frappe UI](https://github.com/frappe/frappe-ui):** A Vue-based UI library providing a modern and user-friendly interface.

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.  Frappe Cloud handles installation, upgrades, monitoring, and support.

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

1.  Ensure you have Docker, docker-compose, and git installed.
2.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```
3.  Access the application at `http://localhost:8000` using the following credentials:
    *   Username: `Administrator`
    *   Password: `admin`

### Local

1.  Install and start `bench`:
    ```bash
    $ bench start
    ```
2.  In a separate terminal, run:
    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Learning and Community Resources

*   [Frappe School](https://frappe.school): Courses on the Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr): Comprehensive documentation for Frappe HR.
*   [User Forum](https://discuss.erpnext.com/):  Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr): Get instant help from other users.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

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