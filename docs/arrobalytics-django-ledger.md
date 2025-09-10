# Django Ledger: Powerful Accounting for Your Django Applications

Django Ledger provides a robust and flexible double-entry accounting engine, seamlessly integrated into the Django framework.  For more information, visit the original repository: [https://github.com/arrobalytics/django-ledger](https://github.com/arrobalytics/django-ledger).

[![Django Ledger Logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)](https://github.com/arrobalytics/django-ledger)

**Key Features:**

*   **Double-Entry Accounting:**  Ensures financial accuracy through a fundamental accounting principle.
*   **Hierarchical Chart of Accounts:** Organize your finances with a structured and customizable chart of accounts.
*   **Financial Statements:** Generate essential financial reports including Income Statements, Balance Sheets, and Cash Flow statements.
*   **Transaction Management:**  Supports Ledgers, Journal Entries, and Transactions.
*   **Order Management:**  Handles Purchase Orders, Sales Orders, Bills, and Invoices.
*   **Financial Calculations:** Provides built-in financial ratio calculations for in-depth analysis.
*   **Multi-Tenancy Support:** Allows you to manage accounting for multiple organizations within a single application.
*   **Import/Export:**  Includes OFX and QFX file import capabilities.
*   **Inventory Management:** Basic inventory support.
*   **Built-in UI:** Integrated Entity Management UI.
*   **Unit of Measure Support**

## Getting Started

*   [Get Started Guide](https://www.djangoledger.com/get-started)
*   [QuickStart Notebook](https://github.com/arrobalytics/django-ledger/blob/develop/notebooks/QuickStart%20Notebook.ipynb)
*   [Documentation](https://django-ledger.readthedocs.io/en/latest/)
*   [Join our Discord](https://discord.gg/c7PZcbYgrc)

## Installation

Django Ledger is designed to be easily integrated into your existing Django project.

1.  **Add to INSTALLED\_APPS:**

    ```python
    INSTALLED_APPS = [
        ...,
        'django_ledger',
        ...,
    ]
    ```

2.  **Add Context Preprocessor:**

    ```python
    TEMPLATES = [
        {
            'OPTIONS': {
                'context_processors': [
                    '...',
                    'django_ledger.context.django_ledger_context'  # Add this line to a context_processors list..
                ],
            },
        },
    ]
    ```

3.  **Run Migrations:**

    ```bash
    python manage.py migrate
    ```

4.  **Include URLs in your project's `urls.py`:**

    ```python
    from django.urls import include, path

    urlpatterns = [
        ...,
        path('ledger/', include('django_ledger.urls', namespace='django_ledger')),
        ...,
    ]
    ```

5.  **Run your server:**

    ```bash
    python manage.py runserver
    ```

6.  Access Django Ledger via your project's URL (e.g., `http://127.0.0.1:8000/ledger`) using your superuser credentials.

## Setting Up for Development

### Using PipEnv (Recommended)

1.  Clone the repository:

    ```bash
    git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
    ```

2.  Install PipEnv (if not already installed):

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

### Using Docker (Alternative)

1.  Navigate to your project directory.
2.  Give executable permissions to `entrypoint.sh`:

    ```bash
    sudo chmod +x entrypoint.sh
    ```

3.  Add host '0.0.0.0' into `ALLOWED_HOSTS` in `settings.py`.
4.  Build the image and run the container:

    ```bash
    docker compose up --build
    ```

5.  Create a Django superuser inside the container (in a separate terminal):

    ```bash
    docker ps  # Get the container ID
    docker exec -it <container_id> /bin/sh
    python manage.py createsuperuser
    ```

6.  Access Django Ledger via `http://0.0.0.0:8000/` in your browser.

## Run Tests

To run the test suite:

```bash
python manage.py test django_ledger
```

## Screenshots

### Dashboards
![django ledger entity dashboard](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_entity_dashboard.png)

### Financial Statements
![balance_sheet_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/BalanceSheetStatement.png)
![income_statement_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/IncomeStatement.png)
![cash_flow_statement_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/CashFlowStatement.png)

### Other Views
![django ledger bill](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_bill.png)
![django ledger invoice](https://us-east-1.linodeobjects.com/public/img/django_ledger_invoice.png)

## Contribute

Contributions are welcome! Please refer to the [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md).

*   **Feature Requests/Bug Reports**: Open an issue in the repository
*   **For software customization, advanced features and consulting services**:
    [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com