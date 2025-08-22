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

## Frappe HR: Open-Source HRMS for Modern Businesses

**Frappe HR is a complete, open-source HR and payroll software solution designed to streamline your HR processes and empower your workforce.** This powerful HRMS (Human Resource Management System) offers a comprehensive suite of modules, making it easy to manage employees from onboarding to offboarding.  [Explore the Frappe HR Repository on GitHub](https://github.com/frappe/hrms).

### Key Features:

*   **Employee Lifecycle Management:** Manage the entire employee journey, from onboarding to offboarding, with features for promotions, transfers, and exit interviews.
*   **Leave and Attendance Tracking:** Configure leave policies, manage holidays, and track attendance with geolocation check-in/out for accurate timekeeping.
*   **Expense Claims and Advances:** Simplify expense management with multi-level approval workflows, seamlessly integrated with ERPNext accounting.
*   **Performance Management:** Track employee goals, align them with key result areas (KRAs), and streamline appraisal cycles for effective performance reviews.
*   **Payroll & Taxation:** Generate accurate payroll, configure tax slabs, manage salary structures, and generate detailed salary slips.
*   **Mobile App:** Empower your employees with the Frappe HR mobile app to apply for leaves, check-in/out, and access employee profiles on the go.

<details open>

<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Under the Hood

*   **[Frappe Framework](https://github.com/frappe/frappe):** The robust Python and JavaScript full-stack web application framework powering Frappe HR.
*   **[Frappe UI](https://github.com/frappe/frappe-ui):**  A Vue.js-based UI library that provides a modern and user-friendly interface.

## Get Started with Frappe HR

### Production Setup

Simplify your deployment with [Frappe Cloud](https://frappecloud.com), a managed hosting platform that handles installation, setup, upgrades, and support.

<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>

### Development Setup

Follow these steps to set up Frappe HR for local development:

#### Docker
```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```
Access `http://localhost:8000` with username `Administrator` and password `admin`.

#### Local
```bash
# Setup bench (follow Frappe Framework installation steps) and start server
$ bench start
# In a separate terminal:
$ bench new-site hrms.local
$ bench get-app erpnext
$ bench get-app hrms
$ bench --site hrms.local install-app hrms
$ bench --site hrms.local add-to-hosts
# Access at http://hrms.local:8080
```

## Learning and Community

*   [Frappe School](https://frappe.school):  Learn the Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr): Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/): Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr): Get instant help from the community.

## Contributing

Contribute to the project by following the guidelines:

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

*   **Clear and Concise Headline:**  "Frappe HR: Open-Source HRMS for Modern Businesses" directly states the product and its key benefit.
*   **One-Sentence Hook:**  The first sentence grabs attention and highlights the core value proposition.
*   **Keyword Integration:**  Uses relevant keywords like "open-source HRMS," "HR and payroll software," and key features throughout the document.
*   **Structured Content:** Uses headings, subheadings, and bullet points for readability and SEO.  This helps search engines understand the content's structure.
*   **Call to Action (CTA):** Links back to the GitHub repository prominently.
*   **Concise Feature Descriptions:**  Each feature is described briefly, highlighting its benefits.
*   **Clear "Getting Started" Section:**  Provides clear instructions for both production and development setups.
*   **Community Links:** Provides links for learning and community engagement.
*   **Markdown Formatting:**  Uses markdown for clear formatting, important for both readability and SEO.
*   **Emphasis on Open Source:** Repeatedly highlights the open-source nature of the software.
*   **Strategic Keyword Use:** Keywords are used naturally throughout the text, optimizing for search.
*   **Improved Readability:** The updated text is more concise and easier to scan.