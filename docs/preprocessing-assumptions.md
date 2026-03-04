# Preprocessing Assumptions

Preprocessing should preserve fault signals that are easy to lose in generic text cleaning.

- Keep device IDs and station names when they identify recurring failure patterns.
- Normalize repeated punctuation without removing urgency markers entirely.
- Preserve numeric ranges such as temperature, pressure, and duration.
- Map obvious synonyms only when they are stable in maintenance language.
- Keep short tickets; short text often carries the most important operator signal.

The cleaning pipeline should be conservative until label quality and downstream error cases are reviewed.
