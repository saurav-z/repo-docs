<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open-Source HR and Payroll Software</h2>
    <p align="center">
        <b>Streamline your HR processes with Frappe HR, the open-source HRMS solution designed for modern businesses.</b>
    </p>

    [![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
    [![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

    <a href="https://trendshift.io/repositories/10972" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10972" alt="frappe%2Fhrms | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<div align="center">
    <img src=".github/hrms-hero.png"/>
</div>

<div align="center">
    <a href="https://frappe.io/hr">Website</a>
    -
    <a href="https://docs.frappe.io/hr/introduction">Documentation</a>
</div>

## About Frappe HR

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to help businesses manage their entire employee lifecycle efficiently.  Built by the creators of ERPNext, Frappe HR boasts over 13 modules, providing a robust solution for managing all aspects of HR, from onboarding to payroll.

**[Explore the Frappe HR repository on GitHub](https://github.com/frappe/hrms)**

## Key Features

*   **Employee Lifecycle Management:**  Handle onboarding, promotions, transfers, and exit interviews, simplifying employee management.
*   **Leave and Attendance Tracking:**  Configure leave policies, track attendance with geolocation, and manage leave balances seamlessly.
*   **Expense Claims and Advances:**  Manage employee advances, expense claims, and automate approval workflows with ERPNext integration.
*   **Performance Management:** Set goals, track key result areas (KRAs), and facilitate performance appraisals.
*   **Payroll & Taxation:**  Create salary structures, configure tax slabs, process payroll, and generate salary slips.
*   **Mobile App:**  Access key HR functions on the go, including leave applications, attendance tracking, and employee profile viewing.

<details open>
  <summary>View Screenshots</summary>
  <img src=".github/hrms-appraisal.png"/>
  <img src=".github/hrms-requisition.png"/>
  <img src=".github/hrms-attendance.png"/>
  <img src=".github/hrms-salary.png"/>
  <img src=".github/hrms-pwa.png"/>
</details>

## Under the Hood

Frappe HR is built on the following robust technologies:

*   [**Frappe Framework**](https://github.com/frappe/frappe):  A full-stack web application framework (Python & Javascript) providing the core foundation for the application.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A modern Vue-based UI library for a responsive and user-friendly interface.

## Production Setup

### Managed Hosting

Consider using [Frappe Cloud](https://frappecloud.com) for hassle-free deployment and management of your Frappe HR instance. It handles installations, upgrades, monitoring, and maintenance.

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

1.  Ensure you have Docker, docker-compose, and Git installed.  Refer to the [Docker documentation](https://docs.docker.com/).
2.  Run the following commands:

```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```

3.  Access the application at `http://localhost:8000` with the following credentials:

    *   Username: `Administrator`
    *   Password: `admin`

### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and keep the server running.

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

## Learning and Community

*   [Frappe School](https://frappe.school): Learn Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr): Comprehensive documentation.
*   [User Forum](https://discuss.erpnext.com/): Engage with the community.
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

*   **Clear Headline and Hook:** The headline now includes the target keywords ("Frappe HR", "Open Source HRMS", "HR and Payroll Software"). The opening sentence immediately tells the user the value proposition.
*   **Keyword Optimization:** The text strategically incorporates relevant keywords (HRMS, HR software, payroll, employee management, etc.) naturally throughout the document.
*   **Bulleted Key Features:** This format makes the main benefits very easy to scan, which is good for both users and search engines.
*   **Subheadings:** Clearly structured sections enhance readability and organization, allowing search engines to understand the content's structure.
*   **Internal & External Links:** Links to relevant resources (documentation, website, GitHub) and key projects (Frappe Framework, Frappe UI) provide context and improve SEO. The backlink to the GitHub repository is emphasized.
*   **Concise Language:** The text is streamlined for clarity and impact.
*   **Mobile-Friendly:** The use of Markdown and appropriate HTML formatting ensures the README is easily readable on mobile devices.
*   **Alt Text:**  Alt text is included on the images for accessibility and SEO.
*   **Call to Action:** Includes a clear call to action, "Explore the Frappe HR repository on GitHub"