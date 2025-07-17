<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HRMS Solution</h2>
	<p align="center">
		<b>Manage your entire employee lifecycle with Frappe HR, a modern, open-source HR and Payroll software.</b>
	</p>
</div>

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

<div align="center">
	<img src=".github/hrms-hero.png" alt="Frappe HR Dashboard"/>
</div>

<div align="center">
	<a href="https://frappe.io/hr">Website</a>
	-
	<a href="https://docs.frappe.io/hr/introduction">Documentation</a>
</div>

## Frappe HR: Your Complete Open-Source HRMS Solution

Frappe HR is a comprehensive, open-source Human Resources Management System (HRMS) designed to streamline and automate your HR processes. With over 13 integrated modules, Frappe HR empowers you to manage your entire employee lifecycle, from onboarding to payroll, all in one place. Built on the robust Frappe Framework, it's a modern and easy-to-use solution for businesses of all sizes.

**[View the Frappe HR Repository on GitHub](https://github.com/frappe/hrms)**

## Key Features of Frappe HR

Frappe HR offers a rich set of features to manage your HR needs effectively:

*   **Employee Lifecycle Management:**  Simplify the employee journey with comprehensive tools for onboarding, promotions, transfers, and exit interviews.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, track attendance with geolocation, and manage leave balances efficiently.
*   **Expense Claims and Advances:**  Manage employee advances and claims with multi-level approval workflows and seamless ERPNext accounting integration.
*   **Performance Management:**  Set and track goals, align with key result areas (KRAs), facilitate employee self-evaluations, and simplify appraisal cycles.
*   **Payroll and Taxation:**  Create salary structures, configure tax slabs, run payroll, handle off-cycle payments, and generate detailed salary slips.
*   **Mobile App:**  Empower your employees with the Frappe HR mobile app for on-the-go leave applications, approvals, and employee profile access.

<details open>
  <summary>View Screenshots</summary>
  <img src=".github/hrms-appraisal.png" alt="Appraisal Screenshot"/>
  <img src=".github/hrms-requisition.png" alt="Requisition Screenshot"/>
  <img src=".github/hrms-attendance.png" alt="Attendance Screenshot"/>
  <img src=".github/hrms-salary.png" alt="Salary Screenshot"/>
  <img src=".github/hrms-pwa.png" alt="PWA Screenshot"/>
</details>

### Under the Hood

*   [**Frappe Framework**](https://github.com/frappe/frappe): A full-stack web application framework written in Python and Javascript. The framework provides a robust foundation for building web applications, including a database abstraction layer, user authentication, and a REST API.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library, to provide a modern user interface. The Frappe UI library provides a variety of components that can be used to build single-page applications on top of the Frappe Framework.

## Production Setup

### Managed Hosting

Experience the ease of Frappe Cloud ([https://frappecloud.com](https://frappecloud.com)), a hassle-free platform for hosting your Frappe applications. Enjoy simplified installation, upgrades, and maintenance, allowing you to focus on your business.

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

1.  Ensure you have Docker, docker-compose, and Git installed. Refer to the [Docker documentation](https://docs.docker.com/).
2.  Run these commands:

    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```

3.  Wait for the setup script to create a site. Access the HR login screen at `http://localhost:8000`.
4.  Use these credentials:

    *   Username: `Administrator`
    *   Password: `admin`

### Local

1.  Set up Bench following the [Installation Steps](https://frappeframework.com/docs/user/en/installation). Start the server:

    ```bash
    $ bench start
    ```

2.  In a new terminal window:

    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```

3.  Access the site at `http://hrms.local:8080`

## Learning and Community

*   [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help from the user community.

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
Key improvements and explanations:

*   **SEO Optimization:** Includes the keywords "HRMS," "Open Source HR," "HR and Payroll Software," and related terms throughout the headings and descriptions.
*   **Clear Hook:** The one-sentence hook immediately grabs the reader's attention.
*   **Structured Headings:** Uses clear headings and subheadings to organize the content.
*   **Bulleted Key Features:** Highlights the main features in an easy-to-read format.
*   **Concise Descriptions:** Provides brief, informative descriptions of each feature.
*   **Call to Action:**  Encourages the user to view the repository.
*   **Community and Learning:** Provides links to resources for users and developers.
*   **Clean Formatting:** Uses Markdown for readability.
*   **Added `alt` attributes** to the images for better SEO and accessibility.
*   **More descriptive wording:** Refined language throughout for better understanding.
*   **Added "alt" text to all images.** This improves accessibility and SEO.
*   **Explicitly mentions Open Source in the title and first paragraph**, emphasizing a key selling point.