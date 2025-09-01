<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open-Source HRMS for Modern Businesses</h2>
    <p>
        <p>Transform your HR processes with Frappe HR, a free and powerful HR and payroll software solution built for efficiency and growth.</p>
    </p>

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

<a href="https://trendshift.io/repositories/10972" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10972" alt="frappe%2Fhrms | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<div align="center">
    <img src=".github/hrms-hero.png" alt="Frappe HR Screenshot"/>
</div>

<div align="center">
    <a href="https://frappe.io/hr">Website</a>
    -
    <a href="https://docs.frappe.io/hr/introduction">Documentation</a>
    -
    <a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## About Frappe HR

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline and automate all aspects of HR operations. From employee onboarding and leave management to payroll and performance tracking, Frappe HR empowers businesses to manage their workforce efficiently.  It's built on the robust Frappe Framework, providing a scalable and customizable solution for organizations of all sizes.

## Key Features

*   **Employee Lifecycle Management:** Manage employees from onboarding to offboarding, including promotions, transfers, and exit interviews.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, track attendance with geolocation, and manage leave balances seamlessly.
*   **Expense Claims and Advances:** Streamline expense claims with multi-level approval workflows and integration with ERPNext accounting.
*   **Performance Management:** Set and track goals, align with key result areas (KRAs), and facilitate employee self-evaluations.
*   **Payroll & Taxation:** Create salary structures, configure tax slabs, process payroll, and generate salary slips.
*   **Mobile App:**  Empower employees with on-the-go access via the Frappe HR mobile app, including leave requests, approvals, and attendance tracking.

<details open>
    <summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png" alt="Appraisal Screenshot"/>
    <img src=".github/hrms-requisition.png" alt="Requisition Screenshot"/>
    <img src=".github/hrms-attendance.png" alt="Attendance Screenshot"/>
    <img src=".github/hrms-salary.png" alt="Salary Screenshot"/>
    <img src=".github/hrms-pwa.png" alt="PWA Screenshot"/>
</details>

## Tech Stack

Frappe HR is built using the following core technologies:

*   **Frappe Framework:**  A full-stack web application framework built in Python and JavaScript providing the foundation for web applications.
*   **Frappe UI:** A Vue.js-based UI library to provide a modern and intuitive user interface.

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. Frappe Cloud takes care of installation, upgrades, monitoring, and support.

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

Requires Docker and docker-compose.

1.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```
2.  Access the application at `http://localhost:8000` using the following credentials:

    *   Username: `Administrator`
    *   Password: `admin`

### Local

1.  Install and start bench, following the [Installation Steps](https://frappeframework.com/docs/user/en/installation).
    ```bash
    bench start
    ```

2.  In a separate terminal window, run the following commands
    ```bash
    bench new-site hrms.local
    bench get-app erpnext
    bench get-app hrms
    bench --site hrms.local install-app hrms
    bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Learning and Community

*   [Frappe School](https://frappe.school) - Learn the Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Extensive documentation for Frappe HR.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help from the community.

## Contributing

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
```
Key improvements and SEO considerations:

*   **Headline Optimization:**  Used a more descriptive and SEO-friendly headline: "Frappe HR: Open-Source HRMS for Modern Businesses". Includes primary keywords.
*   **Concise Introduction:** Started with a strong one-sentence hook that highlights the benefits:  "Transform your HR processes with Frappe HR, a free and powerful HR and payroll software solution built for efficiency and growth."
*   **Keyword Integration:** Weaved relevant keywords (HRMS, HR, payroll, open-source) naturally throughout the description.
*   **Clear Headings and Structure:** Uses clear headings and subheadings to improve readability and SEO ranking.
*   **Detailed Feature Descriptions:** Expanded feature descriptions to provide more context and value.
*   **ALT Text for Images:** Added `alt` text to all images for accessibility and SEO.
*   **Call to Action:** Encouraged exploration of the website and documentation.
*   **GitHub Link:** Added a direct link back to the original repo for easy access and contribution.
*   **Clearer Development Instructions:** Made the Docker and Local setup instructions slightly more user-friendly.
*   **Emphasis on Open Source:** Clearly stated the open-source nature of the software.