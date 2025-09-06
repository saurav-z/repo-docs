<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open-Source HR and Payroll Software</h2>
    <p align="center">
        <b>Revolutionize your HR processes with Frappe HR, the modern, open-source solution built for efficiency and employee satisfaction.</b>
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
    -
    <a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## Frappe HR: Your Complete HRMS Solution

Frappe HR is a comprehensive, open-source Human Resources Management System designed to streamline and automate your HR operations.  It offers over 13 modules, providing a complete solution for managing your employees and all related processes.

## Key Features of Frappe HR

*   **Employee Lifecycle Management:**  Manage every stage of the employee journey, from onboarding to offboarding, including promotions, transfers, and performance feedback.
*   **Leave and Attendance Tracking:**  Configure leave policies, manage attendance with geolocation, and track leave balances effortlessly.
*   **Expense Claims and Advances:**  Simplify expense management with multi-level approval workflows and seamless integration with ERPNext accounting.
*   **Performance Management:**  Set and track employee goals, manage appraisal cycles, and align goals with key result areas (KRAs).
*   **Payroll & Taxation:**  Create salary structures, configure tax slabs, run payroll, and generate salary slips with detailed income breakdowns.
*   **Mobile Accessibility:**  Access key HR functions on the go with the Frappe HR Mobile App, including leave requests, approvals, and attendance tracking.

<details open>
    <summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png"/>
    <img src=".github/hrms-requisition.png"/>
    <img src=".github/hrms-attendance.png"/>
    <img src=".github/hrms-salary.png"/>
    <img src=".github/hrms-pwa.png"/>
</details>

## Technology Under the Hood

*   **Frappe Framework:**  A full-stack web application framework (Python and Javascript) providing the foundation for Frappe HR.
*   **Frappe UI:**  A Vue.js-based UI library, offering a modern and intuitive user interface.

## Production Setup

### Managed Hosting

For simplified deployment and management, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. It handles installation, upgrades, monitoring, and support.

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
2.  Run these commands:

    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```

    Access HR at `http://localhost:8000` with:
    - Username: `Administrator`
    - Password: `admin`

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
3.  Access the site at `http://hrms.local:8080`

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
2.  [Documentation](https://docs.frappe.io/hr) - Extensive documentation for Frappe HR.
3.  [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://t.me/frappehr) - Get instant help from the community.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Logo and Trademark Policy

Please review the [Logo and Trademark Policy](TRADEMARK_POLICY.md).

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
Key improvements and explanations:

*   **SEO Optimization:**  Uses keywords like "open-source HR software," "HRMS," "payroll," and "employee management" in headings and the summary.
*   **One-Sentence Hook:**  A compelling opening sentence to grab the reader's attention.
*   **Clear Headings:**  Uses clear, descriptive headings and subheadings for readability and SEO.
*   **Bulleted Key Features:**  Uses bullet points to highlight the main features, making them easy to scan.
*   **Concise Language:**  Uses more direct and concise language throughout.
*   **Focus on Benefits:**  Emphasizes the benefits of using Frappe HR (e.g., "streamline and automate," "improve employee satisfaction").
*   **GitHub Link:** Added a direct link back to the repo on GitHub.
*   **Improved Formatting:** Consistent formatting and spacing for better readability.
*   **Call to Action (Implied):** Encourages the reader to try the software or learn more.
*   **Comprehensive:** Covers all essential aspects from the original README.
*   **Keyword Integration**:  Naturally incorporates relevant keywords into the text.

This improved README is more informative, more engaging, and better optimized for search engines, while still providing all the essential information about Frappe HR.