# Arrow: Human-Friendly Dates and Times for Python

**Simplify date and time manipulation in Python with Arrow, a powerful library inspired by moment.js, making it easy to create, format, convert, and humanize dates and times.** Explore the original repository: [https://github.com/arrow-py/arrow](https://github.com/arrow-py/arrow)

## Key Features of Arrow

*   **Simplified Date and Time Creation:** Easily create dates and times from various formats.
*   **Timezone Handling:**  Built-in timezone support and UTC as default.
*   **Date/Time Manipulation:** Use the `shift` method for relative offsets (e.g., days, weeks).
*   **Automatic Formatting & Parsing:**  Format and parse strings with ease.
*   **ISO 8601 Support:** Robust support for the widely-used ISO 8601 standard.
*   **Timezone Conversion:** Seamless conversion between timezones.
*   **Date and Time Span/Range generation:** Generate time spans, ranges, floors, and ceilings.
*   **Humanization:** Human-readable date and time representations (e.g., "an hour ago").
*   **Extensible:** Build custom Arrow-derived types.
*   **Type Hints:** Full support for PEP 484-style type hints.

## Why Use Arrow?

Python's built-in date and time modules can be complex and require importing multiple modules. Arrow streamlines date and time operations by:

*   Reducing the number of modules you need to import.
*   Providing a more intuitive and user-friendly API.
*   Offering cleaner timezone handling.
*   Simplifying common tasks like ISO 8601 parsing and humanization.

## Quick Start

**Installation**

Install Arrow using pip:

```bash
pip install -U arrow
```

**Example Usage**

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

# Convert to another timezone
local = utc.to('US/Pacific')
# <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get timestamp
local.timestamp()
# 1368303838.970460

# Format
local.format()
# '2013-05-11 13:23:58 -07:00'

# Format with a specific format
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# '2013-05-11 13:23:58 -07:00'

# Humanize
local.humanize()
# 'an hour ago'
```

## Documentation

For comprehensive documentation, please visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome!  Find an issue, fork the repository, make your changes, add tests, and submit a pull request.  Refer to the original README for detailed contributing instructions.

## Support Arrow

Support Arrow by donating via [Open Collective](https://opencollective.com/arrow).