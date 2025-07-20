<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HR and Payroll Management</h2>
	<p align="center">
		<p>Modern, easy-to-use, and open-source HR and Payroll software designed to streamline your HR processes.</p>
	</p>

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

</div>

<div align="center">
	<img src=".github/hrms-hero.png"/>
</div>

<div align="center">
	<a href="https://frappe.io/hr">Website</a>
	-
	<a href="https://docs.frappe.io/hr/introduction">Documentation</a>
	- <a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## Frappe HR: Your All-in-One HRMS Solution

**Frappe HR** is a comprehensive, open-source Human Resources Management System (HRMS) designed to empower businesses with efficient HR and payroll management.  This robust software offers a wide array of features, from employee lifecycle management to payroll processing and everything in between.

## Key Features of Frappe HR:

*   **Employee Lifecycle Management:**  Handle onboarding, promotions, transfers, and exit interviews, simplifying employee management throughout their journey.
*   **Leave and Attendance Tracking:** Configure leave policies, manage holidays, and track attendance with geolocation features and detailed reports.
*   **Expense Claims and Advances:** Manage employee advances and claims with multi-level approval workflows, seamlessly integrated with ERPNext accounting.
*   **Performance Management:**  Set and track goals, align them with Key Result Areas (KRAs), and facilitate appraisal cycles.
*   **Payroll and Taxation:**  Create salary structures, configure tax slabs, run payroll, accommodate off-cycle payments, and provide detailed salary slips.
*   **Mobile App:**  Access key HR functions on the go with the Frappe HR mobile app, including leave applications, attendance tracking, and employee profile access.

<details open>
<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

## Under the Hood

*   **Frappe Framework:** The foundation, a full-stack web application framework built with Python and JavaScript.  It provides the tools you need to build robust web applications, including a database abstraction layer, user authentication, and a REST API.  [Learn More](https://github.com/frappe/frappe)
*   **Frappe UI:** Provides a modern and user-friendly interface with a Vue-based UI library.  [Learn More](https://github.com/frappe/frappe-ui)

## Production Setup

### Managed Hosting

For a hassle-free experience, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hosting Frappe applications. It handles installation, updates, and support so you can focus on your business.

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

Get started quickly with Docker:

1.  **Prerequisites:** Ensure Docker, docker-compose, and Git are installed.
2.  **Commands:**

```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```

Access the HR application at `http://localhost:8000` using the credentials:

*   Username: `Administrator`
*   Password: `admin`

### Local Setup

Follow these steps to set up Frappe HR locally:

1.  **Bench Setup:**  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up bench and start the server:
    ```bash
    $ bench start
    ```
2.  **New Site Setup:** In a separate terminal:
    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```

Access the site at `http://hrms.local:8080`

## Learning and Community

*   [Frappe School](https://frappe.school): Learn from the experts.
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
Key improvements and optimizations:

*   **SEO-Optimized Title:**  "Frappe HR: Open Source HR and Payroll Management" includes relevant keywords.
*   **One-Sentence Hook:**  Clear and concise introduction.
*   **Bulleted Key Features:**  Improved readability and scannability.
*   **Clear Headings:**  Organized content for easy navigation.
*   **Descriptive Subheadings:**  Enhance understanding of different sections.
*   **Links to GitHub:** Directs users to the source code.
*   **Community & Learning Sections:**  Makes it easy for users to discover more about Frappe HR.
*   **Concise and Informative:**  Improved wording.
*   **Call to Actions**:  Added more links to encourage action.
*   **Emphasis on Open Source:**  Highlights a key selling point.
*   **Docker Instructions Improved**:  More clear instructions.