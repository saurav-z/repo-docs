<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HR and Payroll Software</h2>
	<p align="center">
		<b>Manage your entire employee lifecycle with Frappe HR, the modern, open-source HRMS.</b>
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
    - <a href="https://github.com/frappe/hrms">View on GitHub</a>
</div>

## About Frappe HR

Frappe HR is a comprehensive, **open-source Human Resources Management System (HRMS)** designed to streamline your HR processes. Built on the robust Frappe Framework, it offers a modern, user-friendly interface and a complete suite of tools for managing your employees.  It's perfect for businesses of all sizes looking for a flexible, cost-effective HR solution.

## Key Features of Frappe HR

Frappe HR is packed with features to manage your entire HR lifecycle.

*   **Employee Lifecycle Management**: Easily onboard new hires, manage promotions, transfers, and conduct exit interviews, creating a smooth employee experience.
*   **Leave and Attendance Tracking**:  Configure custom leave policies, automatically pull in regional holidays, utilize geolocation check-in/out, and generate attendance reports.
*   **Expense Claims and Advances**: Manage employee advances, streamline expense claims with multi-level approval workflows, and seamlessly integrate with ERPNext accounting.
*   **Performance Management**: Set and track employee goals, align them with key result areas (KRAs), enable self-evaluations, and simplify appraisal cycles.
*   **Payroll & Taxation**:  Create flexible salary structures, configure tax slabs, run payroll processing, handle off-cycle payments, and provide detailed salary slips.
*   **Mobile App**: Access essential HR functions on the go with the Frappe HR mobile app, including leave requests/approvals, and attendance tracking.

<details open>
<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Under the Hood

*   [**Frappe Framework**](https://github.com/frappe/frappe): The foundation for the HRMS, a full-stack web application framework written in Python and Javascript.
*   [**Frappe UI**](https://github.com/frappe/frappe-ui): A Vue-based UI library providing a modern and intuitive user experience.

## Get Started

### Production Setup

**Frappe Cloud:** For the easiest setup, try [Frappe Cloud](https://frappecloud.com), a managed hosting platform that handles installation, upgrades, and maintenance.

<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

### Development Setup

**Docker:**

1.  Ensure you have Docker and docker-compose installed, along with Git.
2.  Run the following commands in your terminal:

```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```

Access the HRMS at `http://localhost:8000` using the credentials:

*   Username: `Administrator`
*   Password: `admin`

**Local Setup:**

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server and keep it running

    ```sh
    $ bench start
    ```
2.  In a separate terminal window, run the following commands:

    ```sh
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```

3.  Access the site at `http://hrms.local:8080`

## Resources and Community

*   [Frappe School](https://frappe.school): Learn about Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr): Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr): Get instant help from the community.

## Contributing

We welcome contributions! Please review these guidelines:

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## Logo and Trademark Policy

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

*   **Clear Headline:**  Includes a direct title "Frappe HR: Open Source HR and Payroll Software" to immediately convey the software's purpose.
*   **SEO-Optimized Hook:** A compelling one-sentence introduction at the beginning, using keywords like "open-source," "HRMS," and "employee lifecycle."
*   **Keyword Integration:**  Throughout the README, relevant keywords like "HR," "Payroll," "HRMS," "Open Source," and specific feature names are used naturally.
*   **Bulleted Key Features:**  Provides a concise, scannable list of core functionalities.
*   **Clear Structure:**  Uses headings (H2) and subheadings (H3) to organize information for readability and SEO.
*   **Concise Descriptions:** Each section's descriptions are brief and to the point.
*   **Call to Action (CTA):**  The "Get Started" section and links to resources encourage users to explore the software.
*   **Link Back to GitHub:** The primary GitHub repository is easily visible with a link to the original source.
*   **Improved Formatting:** Uses bolding for important terms to highlight key aspects for the reader and SEO.
*   **Mobile-Friendly:** Uses standard markdown for easy readability on any device.