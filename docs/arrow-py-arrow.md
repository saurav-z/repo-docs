# Arrow: Dates and Times Made Easy for Python

**Simplify your Python date and time handling with Arrow, a user-friendly library that makes working with dates, times, and timestamps a breeze.**  [View the project on GitHub](https://github.com/arrow-py/arrow)

Arrow is designed to be a drop-in replacement for Python's built-in `datetime` module, enhancing its functionality and usability. Built with inspiration from Moment.js and Requests, Arrow streamlines common date and time operations, reducing complexity and improving readability.

## Key Features:

*   **Intuitive API:** Simple methods for creating, manipulating, formatting, and converting dates and times.
*   **Timezone Awareness:**  Works with timezones by default, handling conversions and UTC seamlessly.
*   **Flexible Input:**  Parses various input formats, including ISO 8601, with ease.
*   **Time Shifting:**  Easily shift dates and times using relative offsets (e.g., days, weeks, hours).
*   **Formatting Options:**  Format dates and times automatically or with custom formats.
*   **Humanization:**  Human-readable time differences (e.g., "an hour ago," "in 2 days") with support for multiple locales.
*   **Time Span Generation:** Generate time spans, ranges, floors, and ceilings.
*   **Wide Support:** Supports Python 3.8+ and offers PEP 484 type hints.
*   **Integration:**  Works with `dateutil`, `pytz`, and `ZoneInfo`.

## Why Choose Arrow?

Arrow overcomes the complexities of Python's standard `datetime` module, addressing usability issues like:

*   Too many modules to import
*   Complex date/time types to work with
*   Verbose timezone and timestamp conversions
*   Lack of built-in ISO 8601 parsing and other useful features

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
arrow.get('2013-05-11T21:23:58.970460+07:00')
# <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
# <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift time
utc = utc.shift(hours=-1)
# <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')
# <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get timestamp
local.timestamp()
# 1368303838.970460

# Format time
local.format()
# '2013-05-11 13:23:58 -07:00'
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# '2013-05-11 13:23:58 -07:00'

# Humanize
local.humanize()
# 'an hour ago'
local.humanize(locale='ko-kr')
# '한시간 전'
```

## Documentation

For detailed information, visit the official documentation at [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome!  Please review the [contributing guidelines](https://github.com/arrow-py/arrow/blob/master/CONTRIBUTING.md) for information.

1.  Find an issue on the [issue tracker](https://github.com/arrow-py/arrow/issues).
2.  Fork the repository and create a branch.
3.  Add tests.
4.  Run tests and linting checks (e.g., using `tox`).
5.  Submit a pull request.

## Support Arrow

Support the project through the [Open Collective](https://opencollective.com/arrow) for financial contributions.