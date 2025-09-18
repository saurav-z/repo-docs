<p align="center">
  <img src="https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png" alt="Django Ledger Logo" width="200"/>
</p>

# Django Ledger: Robust Double-Entry Accounting for Django

**Django Ledger** is a powerful and flexible open-source accounting engine, bringing comprehensive financial management capabilities directly into your Django applications.  [Explore the project on GitHub](https://github.com/arrobalytics/django-ledger).

## Key Features:

*   **Double-Entry Accounting:** Accurate and reliable financial tracking.
*   **Hierarchical Chart of Accounts:** Organize your finances effectively.
*   **Financial Statements:** Generate Income Statements, Balance Sheets, and Cash Flow statements.
*   **Transaction Management:** Ledgers, Journal Entries, and full Transaction tracking.
*   **Financial Ratio Calculations:**  Gain valuable financial insights.
*   **Multi-Tenancy Support:** Manage finances for multiple entities seamlessly.
*   **Order & Invoice Management:** Create and manage Purchase Orders, Sales Orders, Bills, and Invoices.
*   **OFX & QFX File Import:** Easily import financial data from external sources.
*   **Inventory Management:** Track and manage your inventory levels.
*   **Built-in Entity Management UI:** Manage all of your entities with a simple user interface.
*   **Django Admin Integration:** Seamlessly integrate with Django's admin interface.
*   **Unit of Measure Support:** Use UoMs for product or service cost tracking.

## Getting Started

### Installation

Django Ledger is a Django application and requires a working Django project.

**Steps to Add Django Ledger to Your Project:**

1.  **Install:** Add `django_ledger` to your `INSTALLED_APPS` in `settings.py`.
    ```python
    INSTALLED_APPS = [
        ...,
        'django_ledger',
        ...,
    ]
    ```
2.  **Context Processor:** Add the context processor to your `TEMPLATES` setting.
    ```python
    TEMPLATES = [
        {
            'OPTIONS': {
                'context_processors': [
                    '...',
                    'django_ledger.context.django_ledger_context'
                ],
            },
        },
    ]
    ```
3.  **Migrate:** Run database migrations.
    ```bash
    python manage.py migrate
    ```
4.  **URLs:** Include Django Ledger's URLs in your project's `urls.py`.
    ```python
    from django.urls import include, path

    urlpatterns = [
        ...,
        path('ledger/', include('django_ledger.urls', namespace='django_ledger')),
        ...,
    ]
    ```
5.  **Run:** Start your Django development server.
    ```bash
    python manage.py runserver
    ```
6.  **Access:** Access the Django Ledger app (usually at `http://127.0.0.1:8000/ledger`) and log in with your superuser credentials.

**For a Zero-Config Setup:** Check out the [Django Ledger Starter Template](https://github.com/arrobalytics/django-ledger-starter) for a quick start.

## Setting Up for Development

This section is intended for contributors to the project.

**To Set Up a Development Environment:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
    ```
2.  Install PipEnv:
    ```bash
    pip install -U pipenv
    ```
3.  Create and activate the virtual environment:
    ```bash
    pipenv install
    pipenv shell
    ```
4.  Apply migrations:
    ```bash
    python manage.py migrate
    ```
5.  Create a superuser:
    ```bash
    python manage.py createsuperuser
    ```
6.  Run the development server:
    ```bash
    python manage.py runserver
    ```

### Development with Docker

Follow these steps to set up Django Ledger using Docker:

1.  Navigate to your project directory.
2.  Give executable permissions to `entrypoint.sh`:
    ```bash
    sudo chmod +x entrypoint.sh
    ```
3.  Add host `'0.0.0.0'` to `ALLOWED_HOSTS` in `settings.py`.
4.  Build and run the container:
    ```bash
    docker compose up --build
    ```
5.  Create a Django superuser within the running container:
    ```bash
    docker ps
    docker exec -it <containerId> /bin/sh
    python manage.py createsuperuser
    ```
6.  Access the application at `http://0.0.0.0:8000/`.

### Running Tests

To run the test suite:

```bash
python manage.py test django_ledger
```

## Contribution

Contributions are welcome!  Please review the [Contribution Guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md) before submitting pull requests.

*   **Feature Requests/Bug Reports:** Open an issue in the repository.
*   **Customization/Consulting:**  [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com.

## Screenshots

<!-- Image links are placed here -->
![django ledger entity dashboard](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_entity_dashboard.png)
![django ledger balance sheet](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_income_statement.png)
![django ledger income statement](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_balance_sheet.png)
![django ledger bill](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_bill.png)
![django ledger invoice](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_invoice.png)

## Financial Statements

![balance_sheet_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/BalanceSheetStatement.png)
![income_statement_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/IncomeStatement.png)
![cash_flow_statement_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/CashFlowStatement.png)