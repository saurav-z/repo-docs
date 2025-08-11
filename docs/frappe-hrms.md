<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open Source HRMS and Payroll Software</h2>
    <p align="center">
        <b>Simplify HR management with Frappe HR, the modern and open-source HR and Payroll solution designed to drive excellence within your company.</b>
    </p>
</div>

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)
<a href="https://trendshift.io/repositories/10972" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10972" alt="frappe%2Fhrms | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

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

Frappe HR is a comprehensive, open-source HRMS (Human Resource Management System) solution, providing a robust platform to manage your employees and streamline your HR processes. Built by the team behind ERPNext, Frappe HR offers a modern and user-friendly experience, empowering businesses of all sizes to efficiently handle their HR needs.

## Key Features

*   **Employee Lifecycle Management:** Manage the entire employee journey, from onboarding and promotions to transfers and exit interviews.
*   **Leave and Attendance Tracking:** Configure leave policies, track attendance with geolocation, and manage leave balances effectively.
*   **Expense Claims and Advances:**  Handle employee advances, expense claims, and approval workflows.
*   **Performance Management:** Set and track goals, align with KRAs (Key Result Areas), and conduct appraisal cycles with ease.
*   **Payroll and Taxation:** Generate salary structures, manage income tax, and run payroll efficiently, including off-cycle payments and income breakdowns.
*   **Mobile Accessibility:** Access key features on the go with the Frappe HR mobile app.
    *   Apply for and approve leaves on the go.
    *   Check-in and check-out.
    *   Access employee profiles right from the mobile app.

<details open>
    <summary>View Screenshots</summary>
        <img src=".github/hrms-appraisal.png" alt="Frappe HR Appraisal Screenshot"/>
        <img src=".github/hrms-requisition.png" alt="Frappe HR Requisition Screenshot"/>
        <img src=".github/hrms-attendance.png" alt="Frappe HR Attendance Screenshot"/>
        <img src=".github/hrms-salary.png" alt="Frappe HR Salary Screenshot"/>
        <img src=".github/hrms-pwa.png" alt="Frappe HR PWA Screenshot"/>
</details>

### Under the Hood

*   **Frappe Framework:**  A full-stack web application framework built on Python and JavaScript that provides the foundation. ([Frappe Framework GitHub](https://github.com/frappe/frappe))
*   **Frappe UI:** A modern Vue-based UI library that delivers a responsive and user-friendly interface. ([Frappe UI GitHub](https://github.com/frappe/frappe-ui))

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for a streamlined and reliable hosting experience. It simplifies installation, upgrades, monitoring, and support for your Frappe HR deployments.

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

Requires Docker, docker-compose, and git installed.

```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```

Access at `http://localhost:8000` using the following credentials:

*   Username: `Administrator`
*   Password: `admin`

### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server.
    ```bash
    $ bench start
    ```
2.  In a separate terminal window:
    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  Access at `http://hrms.local:8080`

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn from Frappe Framework and ERPNext courses.
2.  [Documentation](https://docs.frappe.io/hr) - Comprehensive documentation for Frappe HR.
3.  [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://t.me/frappehr) - Get instant support from users.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Logo and Trademark Policy

Please review our [Logo and Trademark Policy](TRADEMARK_POLICY.md).

<br/>
<br/>
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

*   **Clear, Concise Hook:**  The one-sentence hook is at the beginning and is SEO-friendly (includes keywords).
*   **Descriptive Headings:**  Uses clear headings for easy navigation.
*   **Bulleted Key Features:**  Emphasizes key benefits with bullet points.
*   **Keyword Optimization:**  Uses relevant keywords like "open source HRMS," "payroll software," "HR management," etc., throughout the text.
*   **Alt Text for Images:** Added `alt` text to images for accessibility and SEO.
*   **Internal Links:** Added links to relevant documentation and community resources.
*   **External Link to GitHub:** Added a prominent link back to the original repository.
*   **Structured Markdown:** Used proper markdown formatting for headings, lists, and code blocks.
*   **Call to Action (Optional):**  While not included in this revision, you could add a call to action like "Try Frappe HR today!" or "Get started with Frappe HR now!" to encourage usage.
*   **Concise Language:** Improved the original text for better readability and to avoid unnecessary words.
*   **Screen Shot Descriptions:** Added descriptions to the screen shots.