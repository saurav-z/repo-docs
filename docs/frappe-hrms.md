<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open Source HR and Payroll Software</h2>
    <p><strong>Manage your entire employee lifecycle with a modern, easy-to-use HRMS solution.</strong></p>
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

## About Frappe HR

Frappe HR is a comprehensive, open-source HR and Payroll Software designed to streamline and automate your HR processes.  It offers a complete HRMS solution with over 13 modules to cover everything from employee management and onboarding to leaves, payroll, and taxation. Built on the robust Frappe Framework, Frappe HR provides a modern and intuitive interface to drive excellence within your company.  Learn more about the project on [GitHub](https://github.com/frappe/hrms).

## Key Features

*   **Employee Lifecycle Management:**  Efficiently manage the entire employee journey, from onboarding to offboarding, including promotions, transfers, and performance feedback.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, automate holiday calendars, enable geolocation check-ins, and generate detailed attendance reports.
*   **Expense and Advance Management:**  Simplify expense claims and employee advances with multi-level approval workflows and seamless integration with ERPNext accounting.
*   **Performance Management:**  Set and track employee goals, align them with key result areas (KRAs), facilitate self-evaluations, and simplify appraisal cycles.
*   **Payroll & Taxation:**  Create custom salary structures, configure income tax slabs, run payroll efficiently, handle off-cycle payments, and generate detailed salary slips.
*   **Mobile App:**  Empower your employees with the Frappe HR mobile app to apply for leaves, approve requests, and access their profiles on the go.

<details open>
    <summary>View Screenshots</summary>
        <img src=".github/hrms-appraisal.png"/>
        <img src=".github/hrms-requisition.png"/>
        <img src=".github/hrms-attendance.png"/>
        <img src=".github/hrms-salary.png"/>
        <img src=".github/hrms-pwa.png"/>
</details>

### Under the Hood

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework that provides a powerful foundation for building modern web applications.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue.js-based UI library providing a modern, responsive user interface for Frappe applications.

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.  Frappe Cloud handles installation, upgrades, monitoring, and support, making it easy to manage your Frappe HR instance.

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

1.  Ensure you have Docker, docker-compose, and git installed. Refer to the [Docker documentation](https://docs.docker.com/).
2.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```
3.  Access Frappe HR at `http://localhost:8000` using the following credentials:
    *   Username: `Administrator`
    *   Password: `admin`

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

1.  [Frappe School](https://frappe.school) - Comprehensive courses on the Frappe Framework and ERPNext.
2.  [Documentation](https://docs.frappe.io/hr) - Detailed documentation for Frappe HR.
3.  [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://t.me/frappehr) - Get instant support from fellow users.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

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
Key improvements and explanations:

*   **SEO Optimization:**
    *   Included relevant keywords in the title (Frappe HR, Open Source, HR and Payroll Software).
    *   Used descriptive headings (e.g., "Key Features") for better organization and searchability.
    *   Added a concise introductory paragraph that includes keywords.
*   **Hook:**  "Manage your entire employee lifecycle with a modern, easy-to-use HRMS solution."  This immediately tells the user the benefit.
*   **Clear and Concise Language:** Reworded sentences for better readability and impact.
*   **Bulleted Key Features:** Easier to scan and quickly understand the main benefits.
*   **Link to GitHub:**  Provided a direct link to the GitHub repository.
*   **Structured Content:**  Organized the information with clear headings and subheadings for a better user experience.
*   **Call to Action (Implicit):** The feature list and production setup guides implicitly encourage users to try the software.
*   **Benefits-Focused:**  Highlighted the *benefits* of using Frappe HR rather than just listing features.
*   **Emphasis on Open Source:** Made sure to highlight the open-source nature of the project.
*   **Community Resources:**  Expanded on the community resources to improve accessibility.