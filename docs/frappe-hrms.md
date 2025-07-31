<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open Source HR & Payroll Software</h2>
    <p>Empower your business with Frappe HR, the modern and easy-to-use open-source HRMS solution.</p>

    [![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
    [![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

    <a href="https://trendshift.io/repositories/10972" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10972" alt="frappe%2Fhrms | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<div align="center">
    <img src=".github/hrms-hero.png" alt="Frappe HR Hero Image"/>
</div>

<div align="center">
    <a href="https://frappe.io/hr">Website</a>
    -
    <a href="https://docs.frappe.io/hr/introduction">Documentation</a>
    -
    <a href="https://github.com/frappe/hrms">GitHub Repository</a>
</div>

## About Frappe HR

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline your HR processes. This powerful software offers over 13 modules, from employee management and onboarding to payroll, taxation, and more. It's built on the robust [Frappe Framework](https://github.com/frappe/frappe), ensuring scalability and ease of use.

## Key Features of Frappe HR:

*   ✅ **Employee Lifecycle Management:**  Manage the entire employee journey, from onboarding to exit, including promotions, transfers, and feedback.
*   ✅ **Leave and Attendance Tracking:** Configure leave policies, manage attendance with geolocation, and generate reports to track time off.
*   ✅ **Expense Claims and Advances:** Simplify expense reporting with multi-level approval workflows and seamless ERPNext accounting integration.
*   ✅ **Performance Management:** Set goals, align with key result areas (KRAs), enable self-evaluations, and manage appraisal cycles.
*   ✅ **Payroll & Taxation:** Create salary structures, manage income tax slabs, run payroll efficiently, and generate salary slips.
*   ✅ **Mobile Accessibility:**  Use the [Frappe HR Mobile App](https://frappe.io/hr) to apply for and approve leaves, check-in/out, and access employee profiles on the go.

<details open>
<summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png" alt="Frappe HR Appraisal Screen"/>
    <img src=".github/hrms-requisition.png" alt="Frappe HR Requisition Screen"/>
    <img src=".github/hrms-attendance.png" alt="Frappe HR Attendance Screen"/>
    <img src=".github/hrms-salary.png" alt="Frappe HR Salary Screen"/>
    <img src=".github/hrms-pwa.png" alt="Frappe HR PWA Screen"/>
</details>

### Built With:

*   [**Frappe Framework**](https://github.com/frappe/frappe): The full-stack web application framework.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A modern Vue-based UI library.

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for easy and secure hosting of your Frappe applications. It handles installation, upgrades, and maintenance.

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

1.  Ensure you have Docker, docker-compose, and Git installed.
2.  Run the following commands:

    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```

3.  Access the application at `http://localhost:8000` with the credentials:

    *   Username: `Administrator`
    *   Password: `admin`

### Local

1.  Set up Bench: Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation).
2.  Start the server:

    ```bash
    bench start
    ```

3.  In a separate terminal:

    ```bash
    bench new-site hrms.local
    bench get-app erpnext
    bench get-app hrms
    bench --site hrms.local install-app hrms
    bench --site hrms.local add-to-hosts
    ```

4.  Access the site at `http://hrms.local:8080`.

## Learning and Community

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help.

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
Key improvements and SEO considerations:

*   **Clear and Concise Title:**  Includes "Open Source HR & Payroll Software" in the title for better searchability.
*   **One-Sentence Hook:**  A compelling opening sentence that summarizes the value proposition.
*   **Keyword Optimization:**  Uses relevant keywords like "HRMS," "HR," "Payroll," "Open Source," and specific features throughout the text.
*   **Structured Headings:** Uses clear headings (e.g., "About Frappe HR," "Key Features") for readability and SEO benefits.
*   **Bulleted Key Features:**  Uses bullet points to highlight key features, making them easily scannable. Includes a checkmark to improve readability.
*   **Link Back to Original Repo:** Adds a direct link back to the GitHub repository.
*   **Alt Text for Images:** Adds `alt` text to images for accessibility and SEO.
*   **Focus on Benefits:** Emphasizes the benefits of using Frappe HR (e.g., "Empower your business").
*   **Call to Action:**  Encourages the user to explore the features and resources.
*   **Internal Linking:** Includes links to related resources like documentation and the mobile app.
*   **Clear and concise descriptions:** Explains each key feature in short and easy to understand sentences.