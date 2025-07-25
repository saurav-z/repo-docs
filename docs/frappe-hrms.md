<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
</div>

## Frappe HR: Open-Source HR and Payroll Software

**Frappe HR is a comprehensive, open-source HRMS solution designed to streamline your HR processes and empower your team.**

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

<div align="center">
    <img src=".github/hrms-hero.png" alt="Frappe HR Dashboard"/>
</div>

<div align="center">
    <a href="https://frappe.io/hr">Website</a> |
    <a href="https://docs.frappe.io/hr/introduction">Documentation</a> |
    <a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## Key Features of Frappe HR

Frappe HR offers a complete suite of HR modules to manage your entire employee lifecycle, including:

*   **Employee Lifecycle Management:** Onboarding, promotions, transfers, and exit interviews for a smooth employee experience.
*   **Leave and Attendance Tracking:** Configure leave policies, manage attendance with geolocation, and generate insightful reports.
*   **Expense Claims and Advances:** Manage employee advances, track expense claims with multi-level approval workflows, and integrate with ERPNext accounting.
*   **Performance Management:** Track goals, manage appraisal cycles, and enable employees to self-evaluate.
*   **Payroll & Taxation:** Create salary structures, configure income tax slabs, run payroll, and generate salary slips.
*   **Mobile App:** Access HR functionalities on the go, including leave requests, approvals, and attendance tracking.

<details open>
    <summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png" alt="Appraisal Screenshot"/>
    <img src=".github/hrms-requisition.png" alt="Requisition Screenshot"/>
    <img src=".github/hrms-attendance.png" alt="Attendance Screenshot"/>
    <img src=".github/hrms-salary.png" alt="Salary Screenshot"/>
    <img src=".github/hrms-pwa.png" alt="PWA Screenshot"/>
</details>

## Under the Hood

*   **Frappe Framework:** A robust, full-stack web application framework (Python and JavaScript).
*   **Frappe UI:** A modern, Vue-based UI library.

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), an open-source platform for hassle-free Frappe application hosting.

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

1.  **Prerequisites:** Docker, docker-compose, and Git installed.
2.  **Commands:**
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```
3.  Access at `http://localhost:8000` (username: `Administrator`, password: `admin`).

### Local

1.  Set up bench (follow [Installation Steps](https://frappeframework.com/docs/user/en/installation)) and start the server:
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
3.  Access at `http://hrms.local:8080`

## Learning and Community

*   [Frappe School](https://frappe.school): Courses on Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr): Extensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr): Get instant help.

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
Key improvements and SEO considerations:

*   **Clear Headline:**  Uses "Frappe HR: Open-Source HR and Payroll Software" for strong keyword targeting.
*   **Concise Hook:** Immediately highlights the core benefit ("streamline your HR processes").
*   **Keyword Optimization:** Repeated use of relevant keywords like "HR," "Payroll," "HRMS," and "open-source" throughout the description.
*   **Bulleted Key Features:**  Easy-to-scan format for quick understanding. Each point is worded to maximize clarity.
*   **Screenshots with Alt Text:** Adds alt text to improve accessibility and SEO.
*   **Clear Sections and Subheadings:**  Organizes the information logically for readability and SEO ranking.
*   **Links to Key Resources:** Directs users to the website, documentation, and the GitHub repository.
*   **Call to Action (Implied):**  The description encourages users to explore the features and consider the platform.
*   **GitHub Link:**  Included a prominent "View on GitHub" link.