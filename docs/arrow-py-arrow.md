# Arrow: Human-Friendly Dates and Times for Python

**Simplify your Python date and time manipulations with Arrow, a user-friendly library that makes working with dates, times, and timestamps a breeze.** (See the [original repository](https://github.com/arrow-py/arrow) for more details.)

## Key Features

Arrow offers a more intuitive and efficient approach to handling dates and times compared to Python's built-in modules. Here's what makes it stand out:

*   **Drop-in Replacement:** Fully implements the datetime module.
*   **Python 3.8+ Compatibility:** Works seamlessly with modern Python versions.
*   **Timezone Awareness:** Defaults to timezone-aware and UTC for consistent handling.
*   **Simplified Creation:** Easy creation options for common input scenarios.
*   **Flexible Shifting:** `shift` method with support for relative offsets, including weeks.
*   **Automatic Formatting and Parsing:** Intelligent string formatting and parsing.
*   **ISO 8601 Support:** Comprehensive support for the widely-used ISO 8601 standard.
*   **Timezone Conversion:** Effortless timezone conversions.
*   **Integration with Existing Libraries:** Supports `dateutil`, `pytz`, and `ZoneInfo` tzinfo objects.
*   **Time Span Capabilities:** Generates time spans, ranges, floors, and ceilings.
*   **Humanization:** Human-readable date and time representations with locale support.
*   **Extensibility:** Extensible for custom Arrow-derived types.
*   **Type Hinting:** Full support for PEP 484-style type hints.

## Why Use Arrow?

Python's built-in date and time modules can be cumbersome. Arrow addresses these pain points:

*   **Reduced Complexity:** Streamlines the use of multiple modules (datetime, time, etc.).
*   **Simplified Types:** Works with fewer date/time types.
*   **Improved Timezone Handling:** Makes timezone conversions easier and more intuitive.
*   **ISO 8601 Parsing and Time Span Support:** Adds missing features from core libraries.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Create an Arrow object from a string
arrow.get('2013-05-11T21:23:58.970460+07:00')  # Returns: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()  # Returns: <Arrow [2023-10-27T15:30:00.000000+00:00]>

# Shift the time
utc = utc.shift(hours=-1)  # Returns: <Arrow [2023-10-27T14:30:00.000000+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')  # Returns: <Arrow [2023-10-27T07:30:00.000000-07:00]>

# Get the timestamp
local.timestamp()  # Returns: 1698426600.0

# Format the date and time
local.format()  # Returns: '2023-10-27 07:30:00 -07:00'
local.format('YYYY-MM-DD HH:mm:ss ZZ')  # Returns: '2023-10-27 07:30:00 -07:00'

# Humanize the date and time
local.humanize()  # Returns: '8 hours ago'
local.humanize(locale='ko-kr')  # Returns: '8시간 전'
```

## Documentation

Find detailed documentation at [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contribute to Arrow's development:

1.  Find an issue on the [issue tracker](https://github.com/arrow-py/arrow/issues).
2.  Fork the repository.
3.  Create a branch and make your changes, including tests.
4.  Run tests: `tox && tox -e lint,docs` OR `make build39 && make test && make lint`.
5.  Submit a pull request.

Ask questions on the [discussions page](https://github.com/arrow-py/arrow/discussions).

## Support Arrow

Support the project through the [Open Collective](https://opencollective.com/arrow) to make one-time or recurring donations.