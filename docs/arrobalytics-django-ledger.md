<!--  django ledger logo -->
[![Django Ledger Logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)](https://github.com/arrobalytics/django-ledger)

# Django Ledger: Powerful Accounting for Your Django Projects

Django Ledger is a robust, open-source accounting engine that seamlessly integrates with Django, empowering developers to build financially driven applications with ease.

**Key Features:**

*   ✅ **Double-Entry Accounting:** Implement accurate and reliable financial tracking.
*   📊 **Financial Reporting:** Generate comprehensive financial statements like Income Statements, Balance Sheets, and Cash Flow Statements.
*   🏦 **Hierarchical Chart of Accounts:** Organize your finances with a flexible and customizable chart.
*   🧾 **Invoice and Order Management:** Handle Purchase Orders, Sales Orders, Bills, and Invoices efficiently.
*   🔄 **Multi-Tenancy Support:** Build applications that support multiple organizations or entities.
*   ➕ **Transactions & Journal Entries:** Manage ledgers, journal entries, and transactions.
*   📥 **Import Capabilities:** Import data via OFX & QFX file formats.
*   ⚙️ **Built-in UI:** Leverage Django Admin integration and a built-in Entity Management UI.
*   📦 **Inventory Management:** Track and manage your inventory.
*   📐 **Financial Ratios:** Get key insights with built-in financial ratio calculations.
*   📏 **Unit of Measure Support:** Track amounts with associated units.
*   🏦 **Bank Account Information:** Manage bank account details.
*   🚪 **Closing Entries:** Perform closing entries with ease.

## Getting Started

*   [FREE Get Started Guide](https://www.djangoledger.com/get-started)
*   [Join our Discord](https://discord.gg/c7PZcbYgrc)
*   [Documentation](https://django-ledger.readthedocs.io/en/latest/)
*   [QuickStart Notebook](https://github.com/arrobalytics/django-ledger/blob/develop/notebooks/QuickStart%20Notebook.ipynb)

## Installation

Django Ledger is designed to work seamlessly within a Django project.  You'll need a working Django project to get started.

### Add django_ledger to INSTALLED_APPS in you new Django Project.

```python
INSTALLED_APPS = [
    ...,
    'django_ledger',
    ...,
]
```

### Add Django Ledger Context Preprocessor

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

### Perform database migrations:

```shell
python manage.py migrate
```

*   Add URLs to your project's __urls.py__:

```python
from django.urls import include, path

urlpatterns = [
    ...,
    path('ledger/', include('django_ledger.urls', namespace='django_ledger')),
    ...,
]
```

### Run your project:

```shell
python manage.py runserver
```

*   Navigate to Django Ledger root view assigned in your project urlpatterns setting (
    typically http://127.0.0.1:8000/ledger
    if you followed this installation guide).
*   Use your superuser credentials to login.

## Deprecated Behavior (v0.8.0+)

*   **DJANGO_LEDGER_USE_DEPRECATED_BEHAVIOR**:  Set to `True` in your Django settings to temporarily enable deprecated features and legacy behaviors while transitioning.  Defaults to `False`.

## Development Setup

Django Ledger includes a development environment within the `dev_env/` folder.

1.  Clone the repository:

```shell
git clone https://github.com/arrobalytics/django-ledger.git && cd django-ledger
```

2.  Install PipEnv:

```shell
pip install -U pipenv
```

3.  Create and activate the virtual environment:

```shell
pipenv install
pipenv shell
```

4.  Apply database migrations:

```shell
python manage.py migrate
```

5.  Create a superuser:

```shell
python manage.py createsuperuser
```

6.  Run the development server:

```shell
python manage.py runserver
```

## Docker Development Setup

1.  Navigate to your project directory.
2.  Give executable permissions to entrypoint.sh

```shell
sudo chmod +x entrypoint.sh
```
3.  Add host '0.0.0.0' into ALLOWED_HOSTS in settings.py.

4.  Build the image and run the container.

```shell
docker compose up --build
```

5.  Add Django Superuser by running command in seprate terminal

```shell
docker ps
```

Select container id of running container and execute following command

```shell
docker exec -it containerId /bin/sh
```

```shell
python manage.py createsuperuser
```

6.  Navigate to http://0.0.0.0:8000/ on browser.

## Testing

Run tests after setting up your development environment:

```shell
python manage.py test django_ledger
```

## Contribute

We welcome contributions!  Please review our [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md) before submitting pull requests.

*   **Feature Requests/Bug Reports:** Open an issue in the repository
*   **Software Customization & Consulting:** [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com

## Screenshots

*   ![Django Ledger Entity Dashboard](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_entity_dashboard.png)
*   ![Django Ledger Balance Sheet](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_balance_sheet.png)
*   ![Django Ledger Income Statement](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_income_statement.png)
*   ![Django Ledger Bill](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_bill.png)
*   ![Django Ledger Invoice](https://us-east-1.linodeobjects.com/django-ledger/public/img/django_ledger_invoice.png)

## Financial Statements Screenshots

*   ![Balance Sheet](https://django-ledger.us-east-1.linodeobjects.com/public/img/BalanceSheetStatement.png)
*   ![Income Statement](https://django-ledger.us-east-1.linodeobjects.com/public/img/IncomeStatement.png)
*   ![Cash Flow Statement](https://django-ledger.us-east-1.linodeobjects.com/public/img/CashFlowStatement.png)

---

**[Visit the Django Ledger Repository on GitHub](https://github.com/arrobalytics/django-ledger)**