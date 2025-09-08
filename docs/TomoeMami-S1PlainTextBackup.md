# S1 Plain Text Backup: Preserving Stage1st Forum Content

**Are you looking for a way to archive and search through text-based content from the Stage1st (S1) forum?** This repository provides a plain text backup of high-activity threads from the S1 forum, making it easier to access and preserve important discussions.

[View the original repository here: TomoeMami/S1PlainTextBackup](https://github.com/TomoeMami/S1PlainTextBackup)

## Key Features

*   **Comprehensive Backups:** Saves plain text versions of active threads from the Stage1st forum.
*   **Regular Updates:** Captures new threads exceeding a certain activity threshold within a specific timeframe.
*   **Optimized for Search:** Text files are organized to facilitate easy searching of forum content.
*   **Archival Structure:**  Threads are archived based on their creation date.
*   **Rich Text Support:** Supports basic formatting including:
    *   Bold text
    *   Hyperlinks
    *   Image formats (jpg, jpeg, png, gif, tif, webp)
    *   Ratings

## Content Filtering & Storage

*   **Thread Selection:**  The repository primarily focuses on threads that meet a defined activity threshold (e.g., number of replies) within a specified timeframe.
*   **File Organization:** Threads are archived in files with approximately 1500 posts per file.
*   **Retention Policy:** Threads remain in the main archive for 3 days. After 3 days, if there are no new replies, the thread's backup will be moved to the historical archive.
*   **File Size Limitation:** Each text file is kept under 1MB to ensure proper rendering on GitHub.

## Historical Archives

Access to historical forum content is available through the following linked repositories:

*   [2020-2021](https://github.com/TomoeMami/S1PlainTextArchive2021)
*   [2022](https://github.com/TomoeMami/S1PlainTextArchive2022)
*   [2023](https://github.com/TomoeMami/S1PlainTextArchive2023)
*   [2024](https://github.com/TomoeMami/S1PlainTextArchive2024)
*   [2025 to Present](https://github.com/TomoeMami/S1PlainTextArchive2025)

## Tools & Resources

*   **S1Downloader:** For local backup of forum content with images and rich text formatting, consider using [S1Downloader](https://github.com/shuangluoxss/Stage1st-downloader).
*   **COVID-19 Thread Backups:** Backups of specific threads (like the COVID-19 forum) are also available on [gitlab](https://gitlab.com/memory-s1/virus)

## Update Log

*   **February 15, 2024:** The criteria for thread collection was modified to include threads with over 40 replies on the first page within 24 hours. Cache duration extended to 14 days.
*   **February 3, 2024:** Criteria adjusted to include threads with over 40 replies on the first page within 12 hours. Cache duration shortened to 7 days.