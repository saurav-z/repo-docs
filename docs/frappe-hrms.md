<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open Source HR and Payroll Software</h2>
    <p align="center">
        **Manage your entire employee lifecycle with Frappe HR, a modern, open-source HRMS solution.**
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

## Frappe HR: Your Complete HRMS Solution

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline and automate your HR processes. It offers a suite of modules, empowering businesses of all sizes to manage their employees efficiently and effectively.

## Key Features of Frappe HR

*   **Employee Lifecycle Management:** From onboarding to offboarding, manage every stage of the employee journey, including promotions, transfers, and exit interviews.
*   **Leave and Attendance Tracking:** Configure custom leave policies, integrate regional holidays, and track employee attendance with geolocation features.
*   **Expense Claims and Advances:** Manage employee advances and expense claims with multi-level approval workflows, integrated seamlessly with ERPNext accounting.
*   **Performance Management:** Set goals, track key result areas (KRAs), and facilitate employee self-evaluations for streamlined appraisal cycles.
*   **Payroll & Taxation:** Generate accurate payroll, configure tax slabs, manage additional salaries and off-cycle payments, and provide detailed salary slips.
*   **Mobile App:** Access Frappe HR functionalities on the go with the mobile app. Apply for and approve leaves, check-in and check-out, and access employee profiles.

<details open>
    <summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png" alt="Appraisal Screenshot"/>
    <img src=".github/hrms-requisition.png" alt="Requisition Screenshot"/>
    <img src=".github/hrms-attendance.png" alt="Attendance Screenshot"/>
    <img src=".github/hrms-salary.png" alt="Salary Screenshot"/>
    <img src=".github/hrms-pwa.png" alt="PWA Screenshot"/>
</details>

## Technology Stack

*   **Frappe Framework:** The robust, full-stack web application framework built with Python and Javascript that powers Frappe HR.
*   **Frappe UI:** A modern and responsive user interface built with Vue.js, ensuring a seamless user experience.

## Production Setup

### Managed Hosting with Frappe Cloud

Simplify your Frappe HR deployment with [Frappe Cloud](https://frappecloud.com). This user-friendly platform handles installation, upgrades, monitoring, and maintenance, allowing you to focus on your business.

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

1.  Ensure you have Docker and Docker Compose installed on your machine.
2.  Clone the repository:
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```
3.  Access Frappe HR at `http://localhost:8000` using the credentials: Username: `Administrator`, Password: `admin`.

### Local

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up bench and start the server.
2.  In a separate terminal window:
    ```bash
    bench new-site hrms.local
    bench get-app erpnext
    bench get-app hrms
    bench --site hrms.local install-app hrms
    bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`.

## Learn More and Engage with the Community

*   [Frappe School](https://frappe.school): Learn the Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr): Comprehensive documentation for Frappe HR.
*   [User Forum](https://discuss.erpnext.com/): Connect with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr): Get instant help and support.

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

*   **Clear Headline:** Included keywords ("Open Source HR" "HRMS") in the main heading for better searchability.
*   **One-Sentence Hook:**  Added a concise sentence to immediately grab attention and summarize Frappe HR's value.
*   **Detailed Feature List with Benefit-Oriented Descriptions:**  Instead of just listing features, briefly describe *what* the features do to increase user engagement.
*   **Clear Technology Stack:** Highlighted the technologies behind the project to attract technically savvy users.
*   **Call to Action:** Encourages users to learn more and join the community.
*   **GitHub Link:** Incorporated a link to the project's GitHub repository to make it accessible.
*   **Semantic HTML:** Used headings and paragraphs to improve readability and SEO.
*   **Keyword Optimization:**  Used relevant keywords naturally throughout the text (e.g., "HRMS," "open-source," "HR," "payroll").
*   **Image Alt Text:** Added alt text to images for accessibility and SEO.
*   **Concise and Focused:** Removed unnecessary details to make the README more accessible.
*   **Formatting for Readability:** Improved the overall presentation of the content using Markdown formatting.
*   **Summarized the original documentation:** Provided a more concise and readable summary of the original documentation.