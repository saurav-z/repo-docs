<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR</h2>
	<p align="center">
		<p>Open Source, modern, and easy-to-use HR and Payroll Software</p>
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

## Frappe HR: Open-Source HRMS Software for Modern Businesses

**Frappe HR** is a comprehensive, open-source HR and Payroll solution designed to streamline your human resource management. [View the original repository on GitHub](https://github.com/frappe/hrms).

### Key Features

*   **Employee Lifecycle Management:** Effortlessly manage the entire employee journey from onboarding to offboarding, including promotions, transfers, and performance feedback.
*   **Leave and Attendance Tracking:** Configure flexible leave policies, integrate with regional holidays, track check-in/check-out with geolocation, and monitor leave balances.
*   **Expense Claims and Advances:** Manage employee advances, handle expense claims, and implement multi-level approval workflows with seamless integration with ERPNext accounting.
*   **Performance Management:** Track employee goals, align them with key result areas (KRAs), and simplify the appraisal cycle.
*   **Payroll and Taxation:** Create customizable salary structures, configure tax slabs, run payroll, handle additional payments, and generate detailed salary slips.
*   **Mobile App:** Empower employees with the Frappe HR mobile app for leave applications, approvals, check-ins/outs, and accessing employee profiles on the go.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Built With

*   **Frappe Framework:** The robust, full-stack web application framework powering Frappe HR, written in Python and Javascript.
*   **Frappe UI:** A modern, Vue.js-based UI library providing a sleek and intuitive user experience.

## Production Setup

### Managed Hosting

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), the user-friendly platform for hosting Frappe applications, offering installation, updates, monitoring, and support.

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
2.  Clone the repository:

    ```bash
    git clone https://github.com/frappe/hrms
    cd hrms/docker
    docker-compose up
    ```

3.  Access Frappe HR at `http://localhost:8000` with the following credentials:
    *   Username: `Administrator`
    *   Password: `admin`

### Local

1.  Follow the [Installation Steps](https://frappeframework.com/docs/user/en/installation) to set up Bench and start the server.
2.  In a separate terminal:

    ```bash
    bench new-site hrms.local
    bench get-app erpnext
    bench get-app hrms
    bench --site hrms.local install-app hrms
    bench --site hrms.local add-to-hosts
    ```

3.  Access the site at `http://hrms.local:8080`

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

*   **Clear Title and Hook:** "Frappe HR: Open-Source HRMS Software for Modern Businesses" immediately grabs attention and describes the core functionality and target audience.
*   **Key Feature Bullets:** Using concise bullet points makes the feature list easily scannable.  Keywords like "HRMS," "employee lifecycle," "payroll," "attendance," etc. are incorporated.
*   **Keyword Optimization:**  The description uses relevant keywords throughout, increasing search visibility.
*   **Strong Structure with Headings:**  Well-defined headings make the document easy to read and navigate.  This is important for both human readers and search engines.
*   **Emphasis on Open Source:** The repeated mention of "open-source" is a key differentiator and a search term.
*   **Call to Action (Implied):** The description encourages action by highlighting the ease of use and the benefits of the software.
*   **Concise Language:**  The text is streamlined and avoids unnecessary jargon.
*   **Link to Original Repo:**  Added a clear link back to the original repo.
*   **Alt text for Images:**  Ensure images have meaningful `alt` text for SEO.

This revised README is much more appealing, informative, and optimized for search engines, helping more people discover and use Frappe HR.