<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
    <h2>Frappe HR: Open Source HR & Payroll Software</h2>
    <p>
        <b>Manage your entire HR lifecycle with Frappe HR, a modern, open-source solution.</b>
    </p>
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

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline and simplify all aspects of your HR operations. Built on the robust Frappe Framework, it offers a modern and user-friendly interface.  [See the original repository](https://github.com/frappe/hrms).

## Key Features of Frappe HR:

*   ✅ **Employee Lifecycle Management:**  Efficiently onboard employees, manage promotions and transfers, and conduct exit interviews to optimize the employee experience.
*   ✅ **Leave and Attendance Tracking:** Configure leave policies, automate holiday schedules, track employee check-in/check-out with geolocation, and monitor leave balances and attendance with comprehensive reports.
*   ✅ **Expense and Advance Management:**  Manage employee advances, simplify expense claims with multi-level approval workflows, and integrate seamlessly with ERPNext accounting.
*   ✅ **Performance Management:** Track employee goals, align them with Key Result Areas (KRAs), facilitate self-evaluations, and streamline appraisal cycles.
*   ✅ **Payroll & Taxation:**  Create salary structures, configure income tax slabs, run payroll, accommodate off-cycle payments, and provide clear income breakdowns on salary slips.
*   ✅ **Frappe HR Mobile App:**  Empower your team with the convenience of applying for and approving leaves, checking in/out, and accessing employee profiles on the go.

<details open>
<summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png" alt="Frappe HR Appraisal Screenshot"/>
    <img src=".github/hrms-requisition.png" alt="Frappe HR Requisition Screenshot"/>
    <img src=".github/hrms-attendance.png" alt="Frappe HR Attendance Screenshot"/>
    <img src=".github/hrms-salary.png" alt="Frappe HR Salary Screenshot"/>
    <img src=".github/hrms-pwa.png" alt="Frappe HR PWA Screenshot"/>
</details>

## Built With:

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework (Python & Javascript).
*   [**Frappe UI**](https://github.com/frappe/frappe-ui):  A Vue-based UI library.

## Production Setup

### Managed Hosting

Simplify your HRMS deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications.

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

1.  **Prerequisites:** Docker, docker-compose, and Git.
2.  **Clone & Run:**

    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```

3.  Access at `http://localhost:8000` with credentials:

    -   Username: `Administrator`
    -   Password: `admin`

### Local

1.  **Bench Setup:** Follow [Installation Steps](https://frappeframework.com/docs/user/en/installation) to install Bench and start the server.
    ```bash
    $ bench start
    ```
2.  **Setup in a separate terminal window:**
    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  Access at `http://hrms.local:8080`

## Learning and Community

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

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

*   **SEO-Optimized Title & Description:** Includes relevant keywords like "Open Source HR," "HRMS," "Payroll Software" to attract search traffic.  The initial sentence is designed to hook readers.
*   **Clear Structure:** Uses headings and subheadings for easy readability.
*   **Bulleted Key Features:**  Highlights the core benefits and functionalities of Frappe HR, using concise bullet points.  Added emojis for emphasis.
*   **Concise Language:** Simplifies and clarifies the original text.
*   **Links:** Includes links to the original repository, documentation, and community resources to encourage engagement.
*   **Alt Text:** Added `alt` text to all image tags for accessibility and SEO.
*   **Focus on Value Proposition:** The text emphasizes the benefits of using Frappe HR to solve HR challenges.
*   **Clear "Call to Action" (Implicit):**  By highlighting features and providing setup instructions, the README encourages users to try and contribute to the project.
*   **Docker Setup:**  Kept the Docker instructions for easy setup.
*   **Community & Contributing:** Added clear information about community resources and contributing guidelines.
*   **Trademark Policy:**  Included the important link to the trademark policy.
*   **Overall Readability:** The content is now better organized, making it easier for users to understand and evaluate the project.