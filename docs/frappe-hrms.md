<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open-Source HRMS Solution</h2>
    <p align="center">
        **Streamline your HR processes with Frappe HR, a modern, open-source HR and Payroll Software designed for efficiency and ease of use.**
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
    - <a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## Frappe HR: Your Complete HRMS Solution

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to handle all aspects of employee management.  From employee onboarding and leave management to payroll and performance reviews, Frappe HR offers a complete suite of tools to drive excellence within your company.

## Key Features

*   **Employee Lifecycle Management:** Efficiently manage the entire employee journey, from onboarding, promotions, and transfers, to exit interviews.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, integrate regional holidays, utilize geolocation for check-in/out, and generate detailed attendance reports.
*   **Expense Claims and Advances:** Manage employee advances, streamline expense claims with multi-level approval workflows, and integrate seamlessly with ERPNext accounting.
*   **Performance Management:** Track and align employee goals with key result areas (KRAs), enable self-evaluations, and simplify appraisal cycles.
*   **Payroll & Taxation:** Configure salary structures, handle income tax calculations, run payroll efficiently, manage additional payments, and provide clear salary slips.
*   **Frappe HR Mobile App:**  Empower your employees with on-the-go access for leave applications, approvals, check-ins, and employee profile access via our mobile app.

<details open>
    <summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png" alt="Appraisal Screenshot"/>
    <img src=".github/hrms-requisition.png" alt="Requisition Screenshot"/>
    <img src=".github/hrms-attendance.png" alt="Attendance Screenshot"/>
    <img src=".github/hrms-salary.png" alt="Salary Screenshot"/>
    <img src=".github/hrms-pwa.png" alt="PWA Screenshot"/>
</details>

### Under the Hood

*   [**Frappe Framework**](https://github.com/frappe/frappe): A powerful, full-stack web application framework built with Python and JavaScript, providing a robust foundation for building web applications.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A modern Vue.js-based UI library that offers a responsive and user-friendly interface.

## Get Started

### Production Setup

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com), a managed hosting solution for Frappe applications.  Frappe Cloud handles installation, upgrades, monitoring, and maintenance.

<div>
    <a href="https://frappecloud.com/hrms/signup" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Development Setup

#### Docker

1.  Install Docker, docker-compose, and Git.
2.  Clone the repository: `git clone https://github.com/frappe/hrms`
3.  Navigate to the Docker directory: `cd hrms/docker`
4.  Run `docker-compose up`
5.  Access Frappe HR at `http://localhost:8000` (Use Administrator/admin for login)

#### Local Setup

1.  Set up Bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server: `$ bench start`
2.  In a separate terminal window, run the following commands:
    ```sh
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Learning and Community

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext through community courses.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get real-time support from the user community.

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
```
Key improvements and SEO optimizations:

*   **Concise Hook:** A clear one-sentence description at the beginning.
*   **Keyword Rich Headings:** Uses relevant keywords like "Open-Source HRMS," "HR and Payroll Software".
*   **Bulleted Feature List:** Easy-to-scan features that highlight key benefits.
*   **SEO-Friendly Content:** Includes phrases like "employee management," "payroll," "attendance,"  "performance management," and "open-source."
*   **Clear Calls to Action:** Promotes Frappe Cloud.
*   **GitHub Link:** Prominently displays the link back to the original repo.
*   **Alt Text:** Added alt text to images for accessibility and SEO.
*   **Simplified Structure:**  Improved readability and flow.
*   **Concise Language:**  Avoids unnecessary words.