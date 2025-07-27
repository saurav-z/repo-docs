<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
</div>

## Frappe HR: Open-Source HR and Payroll Software for Modern Businesses

Frappe HR is a complete, open-source HRMS solution designed to streamline your HR processes, from employee management to payroll. [Visit the original repository on GitHub](https://github.com/frappe/hrms) to learn more and contribute.

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)
<a href="https://trendshift.io/repositories/10972" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10972" alt="frappe%2Fhrms | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<div align="center">
	<img src=".github/hrms-hero.png"/>
</div>

<div align="center">
	<a href="https://frappe.io/hr">Website</a>
	-
	<a href="https://docs.frappe.io/hr/introduction">Documentation</a>
</div>

### Key Features of Frappe HR:

*   **Employee Lifecycle Management**: Comprehensive tools for onboarding, promotions, transfers, and exit interviews, making employee management simple.
*   **Leave and Attendance Tracking**: Configure leave policies, manage attendance with geolocation, and generate attendance reports.
*   **Expense Claims and Advances**: Manage employee advances and claims with multi-level approval workflows, seamlessly integrated with ERPNext accounting.
*   **Performance Management**: Track goals, align with KRAs, conduct appraisals, and facilitate performance reviews.
*   **Payroll & Taxation**: Design salary structures, calculate taxes, run payroll, manage off-cycle payments, and generate detailed salary slips.
*   **Mobile App**: Access key HR functions on the go, including leave applications, attendance tracking, and employee profile viewing.

<details open>
<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Under the Hood

*   **Frappe Framework**: Built on the robust Frappe Framework, a full-stack web application framework for building web applications.
*   **Frappe UI**: Uses Frappe UI for a modern and user-friendly interface built with Vue.js.

## Production Setup

### Managed Hosting

Consider [Frappe Cloud](https://frappecloud.com) for hassle-free Frappe application hosting. It simplifies installation, upgrades, monitoring, and support.

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

1.  Install Docker and Docker Compose.
2.  Clone the repository: `git clone https://github.com/frappe/hrms`
3.  Navigate to the Docker directory: `cd hrms/docker`
4.  Run: `docker-compose up`
5.  Access the application at `http://localhost:8000` with credentials:  Username: `Administrator`, Password: `admin`

### Local

1.  Set up bench following the [Installation Steps](https://frappeframework.com/docs/user/en/installation)
	and start the server: `$ bench start`
2.  In a new terminal, run these commands:
    ```sh
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  Access the site at `http://hrms.local:8080`

## Learning and Community

1.  [Frappe School](https://frappe.school) - Learn Frappe Framework and ERPNext.
2.  [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
3.  [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
4.  [Telegram Group](https://t.me/frappehr) - Get instant help from the community.

## Contributing

1.  [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
2.  [Report Security Vulnerabilities](https://erpnext.com/security)
3.  [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

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

Key improvements:

*   **SEO Optimization:**  Included relevant keywords like "open-source HR software," "HRMS," and specific features.
*   **Clear Headings:** Uses appropriate headings for better readability and structure.
*   **Concise Summary:**  Starts with a compelling one-sentence hook.
*   **Key Features:**  Uses bullet points for easy scanning.
*   **Improved Formatting:**  Uses markdown for better readability on GitHub.
*   **Link to Original Repo:**  Includes a prominent link back to the original repository.
*   **Organized Content:**  Restructured the content for better flow and clarity.
*   **Removed Redundancy:** Streamlined some sections to avoid repetition.
*   **Focus on Benefits:** Highlights what the software *does* (benefits), not just what it *is*.
*   **Call to Action:** Encourages readers to "learn more and contribute".