# Arrow: Human-Friendly Dates and Times for Python

**Simplify date and time manipulation in Python with Arrow, a powerful library designed for intuitive datetime handling.**  Get started with [the original repo](https://github.com/arrow-py/arrow).

## Key Features of Arrow

Arrow offers a streamlined approach to working with dates and times, providing:

*   **Intuitive API:** Easily create, manipulate, format, and convert dates and times.
*   **Simplified datetime replacement:** A drop-in replacement for Python's `datetime` with enhanced functionality.
*   **Timezone awareness:**  Handles timezones and UTC by default, minimizing timezone-related headaches.
*   **Effortless Creation:** Simple creation options for various input formats.
*   **Flexible Time Shifts:**  Use the `shift` method for relative offsets including weeks.
*   **Automatic Formatting & Parsing:** Automatically formats and parses strings, including comprehensive ISO 8601 support.
*   **Timezone Conversion:** Seamless timezone conversions.
*   **Dateutil, Pytz, and ZoneInfo Support:** Integrates with popular timezone libraries.
*   **Time Span Generation:** Generates time spans, ranges, floors, and ceilings.
*   **Humanization:** Human-friendly date and time representations, with a growing list of locales.
*   **Extensible:** Easily extend Arrow to create your own custom date/time types.
*   **Type Hints:** Full support for PEP 484-style type hints.

## Why Choose Arrow?

Arrow addresses the usability shortcomings of Python's built-in datetime modules:

*   **Reduces complexity:** Fewer modules and types to juggle.
*   **Simplifies timezone management:** Makes timezone conversions less verbose.
*   **Adds missing features:** Includes ISO 8601 parsing, timespans, and humanization.

## Quick Start

### Installation

Install Arrow using pip:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Parse a datetime string
dt = arrow.get('2013-05-11T21:23:58.970460+07:00')
print(dt)  # Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get UTC now
utc = arrow.utcnow()
print(utc)  # Output: <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift time
utc = utc.shift(hours=-1)
print(utc)  # Output: <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to another timezone
local = utc.to('US/Pacific')
print(local)  # Output: <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get timestamp
print(local.timestamp())  # Output: 1368303838.97046

# Format time
print(local.format())  # Output: '2013-05-11 13:23:58 -07:00'
print(local.format('YYYY-MM-DD HH:mm:ss ZZ'))  # Output: '2013-05-11 13:23:58 -07:00'

# Humanize time
print(local.humanize())  # Output: 'an hour ago'
print(local.humanize(locale='ko-kr'))  # Output: '한시간 전'
```

## Documentation

For comprehensive documentation, visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contribute

Contributions are welcome!  If you're interested in contributing:

1.  Explore the [issue tracker](https://github.com/arrow-py/arrow/issues) and find an issue or feature to work on, especially those marked as "good first issue".
2.  Fork the repository and create a branch for your changes.
3.  Add tests to cover your changes.
4.  Run the test suite and linting checks (using `tox` or `make` commands, as shown in the original README).
5.  Submit a pull request.
6.  Ask questions at [here](https://github.com/arrow-py/arrow/discussions).

## Support Arrow

Support the Arrow project by making a financial contribution through [Open Collective](https://opencollective.com/arrow).