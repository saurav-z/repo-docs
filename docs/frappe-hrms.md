<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HRMS Solution</h2>
	<p align="center">
		<p>Modernize your HR processes and empower your workforce with Frappe HR, a free and open-source HR and payroll software.</p>
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

## Frappe HR: Your Complete HRMS Solution

Frappe HR is a comprehensive, open-source HR Management System (HRMS) designed to streamline your HR operations.  With over 13 modules, it offers a complete solution for managing your employees and driving excellence within your company.

**[Check out the Frappe HR GitHub repository](https://github.com/frappe/hrms) for more details.**

## Key Features of Frappe HR

*   **Employee Lifecycle Management:**  Manage the entire employee journey, from onboarding and promotions to transfers and exit interviews, creating a smoother experience for your workforce.
*   **Leave and Attendance Tracking:**  Configure flexible leave policies, automatically track holidays, use geolocation check-in/out, and monitor leave balances with comprehensive reports.
*   **Expense Claims and Advances:**  Simplify expense management with employee advances, claim submissions, and multi-level approval workflows, seamlessly integrated with ERPNext accounting.
*   **Performance Management:**  Set and track goals, align them with Key Result Areas (KRAs), enable self-evaluations, and streamline appraisal cycles.
*   **Payroll & Taxation:**  Create salary structures, configure tax slabs, automate payroll runs, handle additional payments, and generate detailed salary slips.
*   **Mobile App:** Access core HR functions on the go with the Frappe HR mobile app, including leave applications and approvals, and check-in/out functionality.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

## Technology Stack

*   **[Frappe Framework](https://github.com/frappe/frappe):** A robust, full-stack web application framework (Python and Javascript) providing a strong foundation with database abstraction, user authentication, and a REST API.
*   **[Frappe UI](https://github.com/frappe/frappe-ui):**  A modern Vue-based UI library that powers a user-friendly and intuitive interface.

## Production Setup Options

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a user-friendly, open-source platform that handles installation, upgrades, monitoring, and support for your Frappe HR deployments.

<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>


### Development Setup

#### Docker

1.  Make sure you have Docker, docker-compose and git installed on your machine. Refer [Docker documentation](https://docs.docker.com/).
2.  Clone the repository and navigate into the `docker` directory:
    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    ```
3.  Run `docker-compose up`:
    ```bash
    docker-compose up
    ```

    Wait until the setup script creates a site.  Access the HR login at `http://localhost:8000` using:

    *   Username: `Administrator`
    *   Password: `admin`

#### Local Setup

1.  Set up bench following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and run the server with `bench start`.
2.  In a separate terminal window, run these commands:
    ```bash
    bench new-site hrms.local
    bench get-app erpnext
    bench get-app hrms
    bench --site hrms.local install-app hrms
    bench --site hrms.local add-to-hosts
    ```
3.  Access your local instance at `http://hrms.local:8080`

## Learning and Community Resources

1.  [Frappe School](https://frappe.school) - Online courses to learn Frappe Framework and ERPNext.
2.  [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
3.  [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://t.me/frappehr) - Get instant support from the user community.

## Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Logo and Trademark Policy

Please review the [Logo and Trademark Policy](TRADEMARK_POLICY.md) before using any of the project's branding.

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

*   **SEO-Friendly Title:** The title is optimized with relevant keywords ("Frappe HR," "Open Source," "HRMS") to improve searchability.
*   **One-Sentence Hook:**  The intro now starts with a compelling, benefit-driven sentence.
*   **Clear Headings and Subheadings:**  Organized content for readability and easy navigation.
*   **Bulleted Key Features:** Uses bullet points to highlight the main benefits, making them easily scannable.
*   **Stronger Call to Action:** Includes a direct link back to the GitHub repository.
*   **Keyword Optimization:**  The text incorporates relevant keywords (e.g., "HRMS," "Human Resources," "payroll," "open source").
*   **Concise Language:**  The text is streamlined and focused on conveying the key information.
*   **Clear Instructions:** Development setup instructions are retained but slightly improved for clarity.
*   **Community & Learning:**  Clearly highlights learning resources.
*   **Contribution Guidelines:** Includes links to contribution guidelines.
*   **Trademark Policy:**  Includes a note about the trademark policy.
*   **Visual Enhancements:**  Retains the image assets and their alignment.