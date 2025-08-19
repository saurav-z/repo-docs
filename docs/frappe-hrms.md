<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open Source HRMS Software</h2>
    <p align="center">
        **Manage your entire employee lifecycle with Frappe HR, the modern, open-source HR and payroll solution.**
    </p>
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
</div>

## Frappe HR: Your Comprehensive HRMS Solution

Frappe HR is a complete, open-source Human Resources Management System (HRMS) designed to streamline and optimize your HR processes. With over 13 modules, Frappe HR empowers you to manage everything from employee onboarding and leave to payroll and performance.

**[Visit the Frappe HR GitHub Repository](https://github.com/frappe/hrms)**

## Key Features

*   **Employee Lifecycle Management:** Seamlessly manage the entire employee journey, from onboarding to offboarding, including promotions, transfers, and exit interviews.
*   **Leave and Attendance Tracking:** Configure custom leave policies, track attendance with geolocation, manage balances, and view comprehensive reports.
*   **Expense Claims and Advances:** Simplify expense management with multi-level approval workflows and seamless integration with ERPNext accounting.
*   **Performance Management:** Track employee goals, align with key result areas (KRAs), facilitate self-evaluations, and manage appraisal cycles.
*   **Payroll & Taxation:** Generate accurate payroll, configure income tax slabs, handle off-cycle payments, and provide detailed salary slips.
*   **Mobile App:** Access key HR functions on the go with the Frappe HR mobile app, including leave applications, attendance check-in/out, and employee profile access.

<details open>
<summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png"/>
    <img src=".github/hrms-requisition.png"/>
    <img src=".github/hrms-attendance.png"/>
    <img src=".github/hrms-salary.png"/>
    <img src=".github/hrms-pwa.png"/>
</details>

### Under the Hood

*   **Frappe Framework:** A full-stack web application framework (Python & Javascript) providing a robust foundation for web applications.  Provides database abstraction, user authentication, and a REST API.
*   **Frappe UI:**  A modern, Vue-based UI library for building user interfaces within the Frappe Framework.

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.  Frappe Cloud handles installation, updates, monitoring, and support.

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

Requires Docker, docker-compose, and git. Follow these steps:

```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```

Access at `http://localhost:8000` with credentials:
*   Username: `Administrator`
*   Password: `admin`

### Local Setup

1.  Set up bench (follow [Installation Steps](https://frappeframework.com/docs/user/en/installation)) and run:
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

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
2.  [Documentation](https://docs.frappe.io/hr) - Extensive documentation.
3.  [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://t.me/frappehr) - Get instant help.

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
```
Key improvements and SEO considerations:

*   **Clear, concise title and introductory sentence:** The title includes keywords like "Open Source" and "HRMS". The intro provides a quick value proposition.
*   **Keyword Optimization:**  Uses relevant keywords throughout the README (e.g., "HRMS," "Open Source HR," "Payroll," "Employee Management").
*   **Structured Content:** Uses headings (H2, H3), bullet points, and details to improve readability and organization.
*   **Feature-Focused:** Highlights key features with clear descriptions.
*   **Call to Action:**  Includes a direct link back to the repository.
*   **Clear instructions** for setup.
*   **Concise and modern language.**
*   **Emphasis on Benefits:** Focuses on the *benefits* of using Frappe HR.
*   **Community and Support:** Highlights the learning and community aspects, which are important for SEO (signals of active development and support).
*   **Markdown formatting for readability.**
*   **Includes relevant links**