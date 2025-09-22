[![Django Ledger Logo](https://us-east-1.linodeobjects.com/django-ledger/logo/django-ledger-logo@2x.png)](https://github.com/arrobalytics/django-ledger)

# Django Ledger: Robust Double-Entry Accounting for Django

**Django Ledger** empowers your Django applications with a comprehensive financial management system, simplifying complex accounting tasks with its user-friendly API.

[Get Started Guide](https://www.djangoledger.com/get-started) | [Join our Discord](https://discord.gg/c7PZcbYgrc) | [Documentation](https://django-ledger.readthedocs.io/en/latest/) | [QuickStart Notebook](https://github.com/arrobalytics/django-ledger/blob/develop/notebooks/QuickStart%20Notebook.ipynb)

## Key Features for Financial Management

*   **Double-Entry Accounting:** Ensures accurate financial tracking.
*   **Hierarchical Chart of Accounts:** Organize your finances effectively.
*   **Financial Statements:** Generate Income Statements, Balance Sheets, and Cash Flow statements.
*   **Order Management:** Includes Purchase Orders, Sales Orders, Bills, and Invoices.
*   **Financial Ratio Calculations:** Gain valuable insights into financial performance.
*   **Multi-tenancy Support:** Enables managing financials for multiple entities.
*   **Ledgers, Journal Entries & Transactions:** Provides detailed financial record-keeping.
*   **OFX & QFX File Import:** Seamlessly integrate financial data.
*   **Closing Entries:** Automate period-end financial processes.
*   **Inventory Management:** Track and manage your inventory.
*   **Unit of Measures:** Define and manage units for financial tracking.
*   **Bank Account Information:** Store and manage bank account details.
*   **Django Admin Integration:** Leverages Django's admin interface for easy management.
*   **Built-in Entity Management UI:** Manage your financial entities.

## Getting Involved in Django Ledger

We welcome contributions from the community to enhance Django Ledger. Whether you're a seasoned developer or just starting out, there are many ways to get involved.

*   **Feature Requests/Bug Reports:** Open an issue in the [repository](https://github.com/arrobalytics/django-ledger).
*   **Software Customization & Advanced Features:** Contact us for consulting services and custom development: [Contact us](https://www.miguelsanda.com/work-with-me/) or email msanda@arrobalytics.com.
*   **Contribute:** Review our [contribution guidelines](https://github.com/arrobalytics/django-ledger/blob/master/Contribute.md).

### Who Should Contribute?

We're looking for contributors with:

*   Python and Django programming skills.
*   Finance and accounting expertise.
*   A passion for building a robust accounting engine API.

## Installation and Setup

Django Ledger is designed as a Django application. This section provides the steps to add Django Ledger to your project:

### Prerequisites

*   Working knowledge of Django.
*   A working Django project.

### Installation Steps

1.  **Add django\_ledger to INSTALLED\_APPS:**

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
                    'django_ledger.context.django_ledger_context'  # Add this line
                ],
            },
        },
    ]
    ```
3.  **Perform database migrations:**

    ```bash
    python manage.py migrate
    ```
4.  **Add URLs to your project's urls.py:**

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

6.  **Access Django Ledger:** Navigate to the Django Ledger root view (typically `http://127.0.0.1:8000/ledger`).
7.  **Login:** Use your superuser credentials to log in.

### Using Django Ledger with the Starter Template

The easiest way to start is to use the zero-config Django Ledger starter template. See details [here](https://github.com/arrobalytics/django-ledger-starter).

### Deprecated Behavior Setting (v0.8.0+)

Since v0.8.0, the `DJANGO_LEDGER_USE_DEPRECATED_BEHAVIOR` setting controls access to deprecated features.

*   **Default:** `False` (deprecated features are disabled by default)
*   To temporarily use deprecated features during transition, set this to `True` in your Django settings.

## Development Setup

Detailed instructions for setting up a development environment are provided in the original README.

### Using Docker

Docker setup is also available as described in the original README.

## Testing

Run the test suite after setting up your environment.

```bash
python manage.py test django_ledger
```

## Screenshots

[Include all the screenshots from the original README here]