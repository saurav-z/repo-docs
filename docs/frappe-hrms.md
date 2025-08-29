<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open Source HRMS Software</h2>
    <p align="center">
        <b>Manage your entire employee lifecycle with Frappe HR, the open-source, modern, and easy-to-use HR and Payroll Software!</b>
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
    -
    <a href="https://github.com/frappe/hrms"><b>View on GitHub</b></a>
</div>

## What is Frappe HR?

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline and automate your HR processes.  This powerful software offers a complete suite of tools to manage your employees from onboarding to offboarding.  Built by the team behind Frappe, it offers a modern and intuitive interface for all your HR needs.

## Key Features

*   **Employee Lifecycle Management:** Manage employees from onboarding, promotions, and transfers, to exit interviews.
*   **Leave and Attendance Tracking:** Configure leave policies, handle holidays, track check-ins, and generate attendance reports.
*   **Expense Claims and Advances:** Manage employee advances and expense claims with multi-level approval workflows, integrating seamlessly with ERPNext accounting.
*   **Performance Management:** Track goals, align with key result areas (KRAs), facilitate self-evaluations, and streamline appraisal cycles.
*   **Payroll & Taxation:** Configure salary structures, manage income tax, run payroll, and view detailed salary slips.
*   **Mobile App:** Access key HR functions on the go with the Frappe HR mobile app, including leave applications, attendance tracking, and employee profile access.

<details open>
<summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png" alt="Appraisal Screen"/>
    <img src=".github/hrms-requisition.png" alt="Requisition Screen"/>
    <img src=".github/hrms-attendance.png" alt="Attendance Screen"/>
    <img src=".github/hrms-salary.png" alt="Salary Screen"/>
    <img src=".github/hrms-pwa.png" alt="PWA Screen"/>
</details>

## Technical Details

Frappe HR is built on the robust [Frappe Framework](https://github.com/frappe/frappe), a full-stack web application framework written in Python and JavaScript, providing a solid foundation for web applications. It also utilizes [Frappe UI](https://github.com/frappe/frappe-ui), a Vue-based UI library to provide a modern user interface.

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. It handles installation, upgrades, monitoring, and support, allowing you to focus on your business.

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

1.  Ensure Docker, docker-compose, and git are installed.
2.  Run the following commands:

```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```

Access HR at `http://localhost:8000` with the following credentials:

*   Username: `Administrator`
*   Password: `admin`

### Local

1.  Set up bench following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server:
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

Access the site at `http://hrms.local:8080`

## Resources and Community

*   [Frappe School](https://frappe.school) - Learn about the Frappe Framework and ERPNext
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help from the user community.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## License and Trademark

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
```
Key improvements and SEO considerations:

*   **Strong Hook:** The one-sentence hook is placed prominently at the beginning, highlighting the key benefit.
*   **Keyword Optimization:**  Keywords like "open source HRMS," "HR and Payroll Software," and specific HR functions are included naturally throughout.
*   **Clear Headings:**  Uses descriptive headings to structure the information logically.
*   **Bulleted Lists:**  Makes key features easily scannable.
*   **Alt Text for Images:**  Added alt text to all images.
*   **Internal Links:**  Uses "View on GitHub" to link to the original repository (crucial).
*   **Concise Language:**  Streamlines descriptions for better readability.
*   **Emphasis on Benefits:**  Focuses on what users *get* from the software.
*   **Call to Action (Implied):** The descriptions encourage users to explore the features.
*   **Structure and Readability:** Enhanced overall formatting, spacing, and the use of `<b>` tags to make the README more visually appealing.
*   **Title Tag:** Added "Frappe HR: Open Source HRMS Software" in the `<h2>` tag for SEO purposes.