<div align="center">
	<a href="https://frappe.io/hr">
		<img src=".github/frappe-hr-logo.png" height="80px" width="80px" alt="Frappe HR Logo">
	</a>
	<h2>Frappe HR: Open Source HRMS for Modern Businesses</h2>
	<p align="center">
		Frappe HR empowers your business with comprehensive HR and payroll management, offering an open-source solution for efficiency and growth.
	</p>
</div>

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
	-
	<a href="https://github.com/frappe/hrms">GitHub Repository</a>
</div>

---

## **About Frappe HR**

Frappe HR is a powerful, open-source HRMS (Human Resource Management System) designed to streamline and automate your HR processes. It provides a comprehensive solution for managing the entire employee lifecycle, from onboarding to payroll, all in one easy-to-use platform. Built on the robust [Frappe Framework](https://github.com/frappe/frappe), Frappe HR is scalable, customizable, and perfect for businesses of all sizes.

## **Key Features of Frappe HR**

*   ✅ **Employee Lifecycle Management:** Simplify onboarding, manage promotions and transfers, and conduct exit interviews.
*   ✅ **Leave and Attendance Tracking:** Configure leave policies, manage attendance with geolocation, and track leave balances.
*   ✅ **Expense Claims and Advances:** Manage employee advances, claim expenses, and streamline approval workflows.
*   ✅ **Performance Management:** Track goals, align with KRAs (Key Result Areas), and simplify appraisal cycles.
*   ✅ **Payroll & Taxation:** Create salary structures, configure tax slabs, run payroll, and view salary slips.
*   ✅ **Mobile App:** Access HR functions on the go with the Frappe HR mobile app.

<details open>
  <summary>View Screenshots</summary>
    <img src=".github/hrms-appraisal.png"/>
    <img src=".github/hrms-requisition.png"/>
    <img src=".github/hrms-attendance.png"/>
    <img src=".github/hrms-salary.png"/>
    <img src=".github/hrms-pwa.png"/>
</details>

## **Technical Underpinnings**

Frappe HR is built upon the following technologies:

*   **Frappe Framework:** A full-stack web application framework (Python and Javascript) that provides a strong foundation.
*   **Frappe UI:** A Vue-based UI library for a modern and responsive user interface.

## **Getting Started**

### Production Setup

For managed hosting, consider [Frappe Cloud](https://frappecloud.com), a user-friendly platform for hassle-free Frappe application hosting.

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

1.  Ensure Docker, docker-compose, and git are installed.
2.  Run the following commands:

```bash
git clone https://github.com/frappe/hrms
cd hrms/docker
docker-compose up
```

Access the application at `http://localhost:8000` with:

*   Username: `Administrator`
*   Password: `admin`

#### Local

1.  Set up bench by following the [Installation Steps](https://frappeframework.com/docs/user/en/installation) and start the server and keep it running.

    ```bash
    $ bench start
    ```

2.  In a separate terminal window, run the following commands

    ```bash
    $ bench new-site hrms.local
    $ bench get-app erpnext
    $ bench get-app hrms
    $ bench --site hrms.local install-app hrms
    $ bench --site hrms.local add-to-hosts
    ```

3.  Access the site at `http://hrms.local:8080`

## **Learn and Connect**

*   [Frappe School](https://frappe.school) - Educational resources for Frappe Framework and ERPNext.
*   [Documentation](https://docs.frappe.io/hr) - Comprehensive Frappe HR documentation.
*   [User Forum](https://discuss.erpnext.com/) - Engage with the community of ERPNext users.
*   [Telegram Group](https://t.me/frappehr) - Get instant help from the community.

## **Contribute**

*   [Issue Guidelines](https://github.com/frappe/erpnext/wiki/Issue-Guidelines)
*   [Report Security Vulnerabilities](https://erpnext.com/security)
*   [Pull Request Requirements](https://github.com/frappe/erpnext/wiki/Contribution-Guidelines)

## **Logo and Trademark Policy**

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