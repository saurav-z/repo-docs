<h1 align="center">
  <br>
  <img alt="WYGIWYH" title="WYGIWYH" src="./.github/img/logo.png" />
  <br>
  WYGIWYH: Simple Finance Tracking for a Clear Financial Picture
  <br>
</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#translations">Translations</a> •
  <a href="#caveats">Caveats & Warnings</a> •
  <a href="#built-with">Built With</a> •
  <a href="https://github.com/eitchtee/WYGIWYH">Original Repository</a>
</p>

**WYGIWYH** (_What You Get Is What You Have_) is a user-friendly and powerful finance tracker designed for a straightforward, no-budget approach to money management, simplifying your finances and helping you take control of your financial future.

[<img src=".github/img/monthly_view.png" width="18%">](https://github.com/eitchtee/WYGIWYH) [<img src=".github/img/yearly.png" width="18%">](https://github.com/eitchtee/WYGIWYH) [<img src=".github/img/networth.png" width="18%">](https://github.com/eitchtee/WYGIWYH) [<img src=".github/img/calendar.png" width="18%">](https://github.com/eitchtee/WYGIWYH) [<img src=".github/img/all_transactions.png" width="18%">](https://github.com/eitchtee/WYGIWYH)

## <a name="about"></a> About WYGIWYH

Tired of complex budgeting apps? WYGIWYH (pronounced "wiggy-wih") offers a refreshing approach to finance tracking based on a simple principle:

> Use what you earn this month for this month. Savings are tracked but treated as untouchable for future months.

This straightforward philosophy helps you avoid overspending while providing a clear overview of your income and expenses. WYGIWYH was built to address the limitations of spreadsheets and complex budgeting apps, providing a solution with multi-currency support, customizability, and automation capabilities.

## <a name="key-features"></a> Key Features

WYGIWYH simplifies your finances with these core features:

*   ✅ **Unified Transaction Tracking:** Easily record all your income and expenses in one place.
*   ✅ **Multi-Account Support:** Track money across multiple banks, wallets, and investment accounts.
*   ✅ **Multi-Currency Support:** Manage transactions and balances in various currencies.
*   ✅ **Custom Currencies:** Create and track custom currencies, like crypto or reward points.
*   ✅ **Automated Adjustments:** Use customizable rules to automatically modify transactions.
*   ✅ **Built-in DCA Tracker:** Track recurring investments, especially for crypto and stocks.
*   ✅ **API Support:** Integrate with other services for automated transaction synchronization.

## <a name="demo"></a> Demo

Explore WYGIWYH's functionality with our demo: [wygiwyh-demo.herculino.com](https://wygiwyh-demo.herculino.com/)

**Demo Credentials:**

>   **Email:** `demo@demo.com`
>
>   **Password:** `wygiwyhdemo`

**Important:** Demo data is automatically wiped every 24 hours or less. API, Rules, Automatic Exchange Rates, and Import/Export features are disabled in the demo.

## <a name="getting-started"></a> Getting Started

To run WYGIWYH, you'll need [Docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/).

1.  **Set up your Environment**

    ```bash
    # Create a project directory (optional)
    $ mkdir WYGIWYH
    $ cd WYGIWYH

    # Create docker-compose.yml
    $ touch docker-compose.yml
    $ nano docker-compose.yml
    # Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/docker-compose.prod.yml and edit according to your needs

    # Configure your environment variables (.env)
    $ touch .env
    $ nano .env # or any other editor
    # Paste the contents of https://github.com/eitchtee/WYGIWYH/blob/main/.env.example and edit accordingly
    ```

2.  **Run the Application**

    ```bash
    $ docker compose up -d
    ```

3.  **Create an Admin Account** (if you haven't set `ADMIN_EMAIL` and `ADMIN_PASSWORD` in your `.env` file)

    ```bash
    $ docker compose exec -it web python manage.py createsuperuser
    ```

### Running Locally

For local development:

1.  Remove `URL` from your `.env` file.
2.  Set `HTTPS_ENABLED` to `false`.
3.  Leave the default `DJANGO_ALLOWED_HOSTS` (localhost 127.0.0.1 [::1]).

    You can then access the application at `localhost:OUTBOUND_PORT`.
    *   If running behind a service like Tailscale, also add your machine's IP to `DJANGO_ALLOWED_HOSTS`.
    *   When using a non-localhost IP, also add it to `DJANGO_ALLOWED_HOSTS`.

### Unraid

WYGIWYH is available as an Unraid template, provided by [nwithan8](https://github.com/nwithan8) in their [unraid_templates](https://github.com/nwithan8/unraid_templates) repository.

WYGIWYH is also available on the Unraid Store. Ensure you provision your own PostgreSQL (version 15 or up) database.

To create the first user, open the container's console using Unraid's UI (Docker page -> WYGIWYH icon -> `Console`) and type `python manage.py createsuperuser`. You will be prompted for your email and password.

### Environment Variables

Refer to the table in the original README for the complete list of environment variables and their descriptions.

### OIDC Configuration

WYGIWYH supports login via OpenID Connect (OIDC) through `django-allauth`. This allows users to authenticate using an external OIDC provider.

Set the following environment variables to configure OIDC:

*   `OIDC_CLIENT_NAME`: Provider's name (default: `OpenID Connect`).
*   `OIDC_CLIENT_ID`:  Your OIDC provider's Client ID.
*   `OIDC_CLIENT_SECRET`: Your OIDC provider's Client Secret.
*   `OIDC_SERVER_URL`:  Your OIDC provider's base URL.
*   `OIDC_ALLOW_SIGNUP`: Enable/disable automatic account creation (default: `true`).

**Callback URL (Redirect URI):**

Configure your OIDC provider with the following callback URL:

`https://your.wygiwyh.domain/auth/oidc/<OIDC_CLIENT_NAME>/login/callback/`

Replace `https://your.wygiwyh.domain` with your WYGIWYH instance URL and `<OIDC_CLIENT_NAME>` with the slugfied value set in OIDC_CLIENT_NAME or the default `openid-connect` if you haven't set this variable.

## <a name="how-it-works"></a> How It Works

For detailed information, explore our [Wiki](https://github.com/eitchtee/WYGIWYH/wiki).

## <a name="translations"></a> Translations

Help us translate WYGIWYH!  [Contribute here](https://translations.herculino.com/engage/wygiwyh/) (log in with your GitHub account).

<a href="https://translations.herculino.com/engage/wygiwyh/">
<img src="https://translations.herculino.com/widget/wygiwyh/open-graph.png" alt="Translation status" />
</a>

## <a name="caveats"></a> Caveats & Warnings

*   I'm not an accountant; some terms and calculations might be simplified. Please report any issues!
*   Most calculations are performed at runtime, which can impact performance.
*   WYGIWYH is not a budgeting or double-entry accounting application.  If these are essential features for you, please open a discussion.

## <a name="built-with"></a> Built With

WYGIWYH is built using a collection of amazing open-source tools:

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