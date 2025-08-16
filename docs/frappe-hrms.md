<div align="center">
    <a href="https://frappe.io/hr">
        <img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
    </a>
</div>

## Frappe HR: Open-Source HRMS Software for Modern Businesses

**Frappe HR is an open-source, comprehensive Human Resources Management System (HRMS) designed to streamline your HR processes and empower your workforce.** Find the original project on [GitHub](https://github.com/frappe/hrms).

<div align="center">
	<img src=".github/hrms-hero.png"/>
</div>

[![CI](https://github.com/frappe/hrms/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/frappe/hrms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frappe/hrms/branch/develop/graph/badge.svg?token=0TwvyUg3I5)](https://codecov.io/gh/frappe/hrms)

<a href="https://trendshift.io/repositories/10972" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10972" alt="frappe%2Fhrms | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<div align="center">
	<a href="https://frappe.io/hr">Website</a>
	-
	<a href="https://docs.frappe.io/hr/introduction">Documentation</a>
</div>


### Key Features

*   **Employee Lifecycle Management:** Manage the complete employee journey, from onboarding to offboarding, including promotions, transfers, and performance feedback.
*   **Leave and Attendance Tracking:** Configure leave policies, track attendance with geolocation, and manage leave balances efficiently.
*   **Expense Claims and Advances:** Handle employee advances and expense claims with multi-level approval workflows.
*   **Performance Management:** Set goals, align them with key result areas (KRAs), and facilitate employee self-evaluations.
*   **Payroll and Taxation:** Generate salary structures, manage income tax, run payroll, and create salary slips.
*   **Mobile App:** Access essential HR functions on the go with the Frappe HR mobile app, including leave applications, attendance tracking, and employee profile access.

<details open>
<summary>View Screenshots</summary>
	<img src=".github/hrms-appraisal.png"/>
	<img src=".github/hrms-requisition.png"/>
	<img src=".github/hrms-attendance.png"/>
	<img src=".github/hrms-salary.png"/>
	<img src=".github/hrms-pwa.png"/>
</details>

### Built with:

*   **Frappe Framework:** A full-stack web application framework for building robust web applications.
*   **Frappe UI:** A modern, Vue-based UI library for a responsive user experience.

### Production Setup

*   **Frappe Cloud:** Host your Frappe applications with ease on Frappe Cloud, a managed platform that handles installation, upgrades, and maintenance.

<div>
	<a href="https://frappecloud.com/hrms/signup" target="_blank">
		<picture>
			<source media="(prefers-color-scheme: dark)" srcset="https://frappe.io/files/try-on-fc-white.png">
			<img src="https://frappe.io/files/try-on-fc-black.png" alt="Try on Frappe Cloud" height="28" />
		</picture>
	</a>
</div>


### Development Setup

Choose from Docker or Local Setup options for your development environment.

**Docker:**

```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```

Access at `http://localhost:8000` with username `Administrator` and password `admin`.

**Local:**

1.  Set up bench following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server.
    ```sh
    $ bench start
    ```
2.  In a separate terminal:
    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```
3.  Access at `http://hrms.local:8080`

### Learning and Community

*   [Frappe School](https://frappe.school) - Learn from courses by the maintainers and community.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive documentation for Frappe HR.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the ERPNext community.
*   [Telegram Group](https://t.me/frappehr) - Get instant help from users.

### Contributing

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

### Logo and Trademark Policy

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