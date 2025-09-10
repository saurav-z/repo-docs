<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open Source HR and Payroll Software</h2>
    <p><b>Manage your entire employee lifecycle with ease using Frappe HR, a modern, open-source HRMS solution.</b></p>
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
</div>

## About Frappe HR

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline and automate your HR processes.  Built by the team behind ERPNext, Frappe HR offers a modern and user-friendly experience, empowering businesses to effectively manage their workforce.  It features over 13 integrated modules to manage the entire employee lifecycle.

[Learn more and explore the code on GitHub](https://github.com/frappe/hrms)

## Key Features

*   ✅ **Employee Lifecycle Management:** From onboarding and promotions to performance reviews and offboarding, manage every stage of the employee journey.
*   ✅ **Leave and Attendance Tracking:** Configure leave policies, manage attendance, and generate reports with ease. Includes features such as geolocation check-in/out.
*   ✅ **Expense Claims and Advances:** Streamline expense claims and employee advance requests with multi-level approval workflows.  Integrates seamlessly with ERPNext accounting.
*   ✅ **Performance Management:** Track goals, align them with key result areas (KRAs), and simplify the appraisal cycle.
*   ✅ **Payroll and Taxation:** Create salary structures, manage tax calculations, run payroll, and generate salary slips.
*   ✅ **Mobile App:** Manage and approve leaves, check-in/out, and access employee profiles on the go via the Frappe HR mobile app.

<details open>
    <summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png" alt="Frappe HR Appraisal Screenshot"/>
    <img src=".github/hrms-requisition.png" alt="Frappe HR Requisition Screenshot"/>
    <img src=".github/hrms-attendance.png" alt="Frappe HR Attendance Screenshot"/>
    <img src=".github/hrms-salary.png" alt="Frappe HR Salary Screenshot"/>
    <img src=".github/hrms-pwa.png" alt="Frappe HR PWA Screenshot"/>
</details>

### Technology Stack

*   **Frappe Framework:** A robust, full-stack web application framework built with Python and JavaScript, providing the foundation for Frappe HR.
*   **Frappe UI:** A modern, Vue.js-based UI library that provides a sleek and user-friendly interface.

## Getting Started

### Production Setup

For easy hosting and management, consider [Frappe Cloud](https://frappecloud.com). Frappe Cloud takes care of installation, upgrades, monitoring, and support for your Frappe deployments.

<div>
    <a href="https://frappecloud.com/hrms/signup" target="_blank">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
            <img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
        </picture>
    </a>
</div>

### Development Setup (Docker)

1.  **Prerequisites:** Docker, docker-compose, and Git.
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```
3.  Access the application at `http://localhost:8000` using the following credentials:

    *   Username: `Administrator`
    *   Password: `admin`

### Development Setup (Local)

1.  Follow the [Frappe Framework Installation Guide](https://frappeframework.com/docs/user/en/installation) to set up bench and start the server.
    ```bash
    bench start
    ```
2.  In a separate terminal, run these commands:
    ```bash
    bench new-site hrms.local
    bench get-app erpnext
    bench get-app hrms
    bench --site hrms.local install-app hrms
    bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Resources & Community

*   [Frappe School](https://frappe.school): Learn the Frappe Framework and ERPNext through community-led courses.
*   [Documentation](https://docs.frappe.io/hr): Comprehensive documentation for Frappe HR.
*   [User Forum](https://discuss.erpnext.com/): Engage with the active ERPNext community.
*   [Telegram Group](https://t.me/frappehr): Get real-time support from the Frappe HR user community.

## Contributing

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
Key improvements and SEO optimizations:

*   **Concise Hook:** Added a strong, one-sentence introduction to immediately grab attention.
*   **Keyword Optimization:** Included relevant keywords like "open source HR," "HRMS," "HR and Payroll," "employee management," etc. throughout the text, especially in headings and the introductory paragraph.
*   **Clear Structure:** Improved headings and organization for better readability and SEO.
*   **Benefit-Driven Language:** Focused on the *benefits* of using Frappe HR (e.g., "streamline HR processes," "manage your workforce effectively").
*   **Bulleted Key Features:**  Used bullet points for easy skimming and better readability.
*   **Alt Text for Images:** Added `alt` text to all images for accessibility and SEO.
*   **Clearer Calls to Action:** Encouraged exploration of the project with a direct link back to the GitHub repo.
*   **Emphasis on "Open Source":** Repeatedly highlighted the "open source" nature of the software to attract the target audience.
*   **Updated Formatting and Visuals**: Improved presentation.