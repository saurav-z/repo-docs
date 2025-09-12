[![Django Ledger Logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)](https://github.com/arrobalytics/django-ledger)

# Django Ledger: A Powerful Double-Entry Accounting Engine for Django

**Django Ledger** provides a robust and flexible solution for incorporating comprehensive financial management features into your Django applications.

[Get Started Guide](https://www.djangoledger.com/get-started) | [Join our Discord](https://discord.gg/c7PZcbYgrc) | [Documentation](https://django-ledger.readthedocs.io/en/latest/) | [QuickStart Notebook](https://github.com/arrobalytics/django-ledger/blob/develop/notebooks/QuickStart%20Notebook.ipynb)

## Key Features

*   **Double-Entry Accounting:** Ensures accurate financial tracking.
*   **Hierarchical Chart of Accounts:** Organizes your finances effectively.
*   **Financial Statements:** Generate Income Statements, Balance Sheets, and Cash Flow Statements.
*   **Transaction Management:** Supports ledgers, journal entries, and comprehensive transaction tracking.
*   **Order and Invoice Management:** Includes Purchase Orders, Sales Orders, Bills, and Invoices.
*   **Financial Ratio Calculations:** Provides valuable insights into your financial performance.
*   **Multi-tenancy Support:** Enables the management of multiple entities within a single application.
*   **Import/Export:** Supports OFX & QFX file import for easy data integration.
*   **Inventory Management:** Includes Unit of Measures for detailed tracking.
*   **Django Admin Integration:** Seamlessly integrates with the Django Admin interface.
*   **Built-in Entity Management UI:** simplifies organization and access control.

## Installation

Django Ledger is a Django application. Ensure you have a working Django project set up before installation. A good place to start is [here](https://docs.djangoproject.com/en/4.2/intro/tutorial01/#creating-a-project).

### Adding Django Ledger to an Existing Project

1.  **Add `django_ledger` to `INSTALLED_APPS`:**

    ```python
    INSTALLED_APPS = [
        ...,
        'django_ledger',
        ...,
    ]
    ```

2.  **Add Django Ledger Context Preprocessor:**

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

3.  **Perform Database Migrations:**

    ```bash
    python manage.py migrate
    ```

4.  **Add URLs to your project's `urls.py`:**

    ```python
    from django.urls import include, path

    urlpatterns = [
        ...,
        path('ledger/', include('django_ledger.urls', namespace='django_ledger')),
        ...,
    ]
    ```

5.  **Run your project:**

    ```bash
    python manage.py runserver
    ```

6.  **Access Django Ledger:** Navigate to the root view (typically `http://127.0.0.1:8000/ledger` if you followed the installation guide) and use your superuser credentials to log in.

### Utilizing Deprecated Behavior (v0.8.0+)

Starting from v0.8.0, Django Ledger introduces the `DJANGO_LEDGER_USE_DEPRECATED_BEHAVIOR` setting:

*   **Default:** `False` (deprecated features are disabled).
*   To temporarily use deprecated features, set this to `True` in your Django settings while you migrate.

## Setting up Django Ledger for Development

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
    ```

2.  **Install PipEnv (if not already installed):**

    ```bash
    pip install -U pipenv
    ```

3.  **Create and activate a virtual environment:**

    ```bash
    pipenv install
    pipenv shell
    ```

    (If you need to specify a Python version: `pipenv install --python PATH_TO_INTERPRETER`)

4.  **Apply migrations:**

    ```bash
    python manage.py migrate
    ```

5.  **Create a superuser:**

    ```bash
    python manage.py createsuperuser
    ```

6.  **Run the development server:**

    ```bash
    python manage.py runserver
    ```

## Setting Up Django Ledger for Development using Docker

1.  **Navigate to your project directory.**

2.  **Give executable permissions to `entrypoint.sh`:**

    ```bash
    sudo chmod +x entrypoint.sh
    ```

3.  **Add host '0.0.0.0' into ALLOWED_HOSTS in settings.py.**

4.  **Build and run the container:**

    ```bash
    docker compose up --build
    ```

5.  **Create a Django superuser:**

    ```bash
    docker ps
    docker exec -it <containerId> /bin/sh
    python manage.py createsuperuser
    ```

6.  **Access the application:** Open your browser and go to `http://0.0.0.0:8000/`.

## Run Test Suite

```bash
python manage.py test django_ledger
```

## Screenshots

### Dashboards
![django ledger entity dashboard](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_entity_dashboard.png)

### Financial Statements
![django ledger balance sheet](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_income_statement.png)
![django ledger income statement](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_balance_sheet.png)

### Transactions
![django ledger bill](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_bill.png)
![django ledger invoice](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_invoice.png)

### Financial Statements Screenshots

![balance_sheet_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/BalanceSheetStatement.png)
![income_statement_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/IncomeStatement.png)
![cash_flow_statement_report](https://django-ledger.us-east-1.linodeobjects.com/public/img/CashFlowStatement.png)

## Getting Involved

We welcome contributions!

*   **Feature Requests/Bug Reports:** Open an issue in the [repository](https://github.com/arrobalytics/django-ledger).
*   **For customization, advanced features, and consulting:** [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com
*   **Contribute:** See our [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md)

## Who Should Contribute?

We are looking for contributors with:

*   Python and Django programming skills
*   Finance and accounting expertise
*   Interest in developing a robust accounting engine API

**[Back to the Top](https://github.com/arrobalytics/django-ledger)**