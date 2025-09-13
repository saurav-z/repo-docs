# Arrow: Python's Human-Friendly Date and Time Library

**Arrow simplifies working with dates and times in Python, making complex operations easy and intuitive.** Developed by [arrow-py](https://github.com/arrow-py/arrow), this library offers a more sensible and human-friendly approach to date and time manipulation.

## Key Features:

*   **Simplified DateTime Replacement:** A drop-in replacement for the Python `datetime` module.
*   **Python 3.8+ Compatibility:** Supports the latest Python versions.
*   **Timezone Awareness:** Defaults to timezone-aware and UTC.
*   **Intuitive Creation:** Easy-to-use creation methods for common date and time formats.
*   **Flexible Shifting:** Supports relative offsets with the `shift` method.
*   **Automated Formatting & Parsing:** Automatically formats and parses strings.
*   **ISO 8601 Support:** Extensive support for the ISO 8601 standard.
*   **Timezone Conversion:** Seamless timezone conversions.
*   **Integration with tzinfo Objects:** Compatible with `dateutil`, `pytz`, and `ZoneInfo`.
*   **Time Span & Range Generation:** Generates time spans, ranges, floors, and ceilings.
*   **Humanization:** Human-readable date and time representations with locale support.
*   **Extensibility:**  Easily extendable for custom date/time types.
*   **Type Hints:** Full support for PEP 484-style type hints.

## Why Choose Arrow?

Arrow addresses the usability issues of Python's standard date and time modules, offering:

*   Fewer modules to import.
*   Fewer data types to manage.
*   Simplified timezone and timestamp conversions.
*   Timezone-aware by default.
*   Built-in functionality for common tasks like ISO 8601 parsing and humanization.

## Quick Start

**Installation:**

```bash
pip install -U arrow
```

**Example Usage:**

```python
import arrow

# Create an Arrow object
utc = arrow.utcnow()
print(utc) # <Arrow [2023-11-14T14:30:00.123456+00:00]>

# Shift the time
local = utc.shift(hours=-1).to('US/Pacific')
print(local) # <Arrow [2023-11-14T06:30:00.123456-08:00]>

# Format the time
print(local.format('YYYY-MM-DD HH:mm:ss ZZ')) # 2023-11-14 06:30:00 -08:00

# Humanize the time
print(local.humanize()) # 'an hour ago'
```

## Resources

*   **Documentation:** [arrow.readthedocs.io](https://arrow.readthedocs.io)
*   **Contribute:** Help improve Arrow by contributing code, documentation, or localizations.  See the [issue tracker](https://github.com/arrow-py/arrow/issues) for opportunities.
*   **Support:** Consider supporting Arrow through the [Open Collective](https://opencollective.com/arrow).