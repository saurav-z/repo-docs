<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Your Powerful, Opinionated Finance Tracker
  <br>
</h1>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#why-wygiwyh">Why WYGIWYH?</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#translate">Help Translate</a> •
  <a href="#caveats-and-warnings">Caveats</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">View on GitHub</a>
</p>

**Tired of complicated budgeting apps? WYGIWYH (What You Get Is What You Have) offers a refreshingly simple and flexible approach to finance tracking, focusing on straightforward money management.**  ([Back to Top](#wygiwyh-your-powerful-opinionated-finance-tracker))

<img src=".github/img/monthly_view.png" width="18%"> <img src=".github/img/yearly.png" width="18%"> <img src=".github/img/networth.png" width="18%"> <img src=".github/img/calendar.png" width="18%"> <img src=".github/img/all_transactions.png" width="18%">

## Overview

WYGIWYH is a powerful, principles-first finance tracker designed for those who prefer a no-budget, straightforward approach to managing their money. It emphasizes simplicity and flexibility, making it easy to track your income and expenses without the constraints of traditional budgeting.

## Key Features

*   **Unified Transaction Tracking:**  Keep all your income and expenses organized in one place.
*   **Multi-Account Support:**  Track funds across various accounts like banks, wallets, and investments.
*   **Multi-Currency Support:**  Manage transactions and balances dynamically in different currencies.
*   **Custom Currencies:**  Define your own currencies for crypto, rewards points, or other custom models.
*   **Automated Adjustments:**  Utilize customizable rules to automatically modify transactions.
*   **Dollar-Cost Averaging (DCA) Tracker:**  Monitor recurring investments, particularly useful for crypto and stocks.
*   **API Support:**  Integrate seamlessly with other services to automate and synchronize transactions.

## Getting Started

To get started with WYGIWYH, you'll need to have [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/) installed.

1.  **Create a Project Directory**
    ```bash
    mkdir WYGIWYH
    cd WYGIWYH
    ```
2.  **Create docker-compose.yml**
    ```bash
    touch docker-compose.yml
    nano docker-compose.yml
    # Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml and edit according to your needs
    ```
3.  **Create .env**
    ```bash
    touch .env
    nano .env # or any other editor you want to use
    # Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/.env.example and edit accordingly
    ```
4.  **Run the Application**
    ```bash
    docker compose up -d
    ```
5.  **Create the First Admin Account**
    ```bash
    docker compose exec -it web python manage.py createsuperuser
    ```
   (If you've set environment variables `ADMIN_EMAIL` and `ADMIN_PASSWORD`, this step is not required.)

For detailed instructions and environment variable configuration, refer to the original [README](https://github.com/eitchtee/WYGIWYH).

## Why WYGIWYH?

WYGIWYH is built on the principle of simplicity: "Use what you earn this month for this month."  This approach helps you manage your finances without the restrictions of budgeting. Learn more about the philosophy in the original [README](https://github.com/eitchtee/WYGIWYH).

## How It Works

Explore the inner workings of WYGIWYH on the [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## Translate

Help translate WYGIWYH! Contribute to the project's internationalization via the [translation platform](https://translations.herculino.com/engage/wygiwyh/).

## Caveats and Warnings

*   **Not an Accountant:** WYGIWYH is not designed or endorsed by accountants, financial advisors, or other financial professionals.
*   **Performance Considerations:**  Calculations are performed at runtime, which can affect performance with large datasets.
*   **Limited Scope:** This is not a budgeting or double-entry accounting application.

## Built With

WYGIWYH is built using these amazing open-source tools:

*   Django
*   HTMX
*   _hyperscript
*   Procrastinate
*   Bootstrap
*   Tailwind
*   Webpack
*   PostgreSQL
*   Django REST framework
*   Alpine.js

[Back to Top](#wygiwyh-your-powerful-opinionated-finance-tracker)