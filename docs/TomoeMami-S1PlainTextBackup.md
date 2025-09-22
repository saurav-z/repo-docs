# S1 Forum Plain Text Backup: Preserving Stage1st Forum Discussions

This repository provides a valuable archive of Stage1st (S1) forum discussions in plain text format, making it easy to search and analyze the content. [View the original repository here](https://github.com/TomoeMami/S1PlainTextBackup).

## Key Features

*   **Comprehensive Archive:** Backups of active S1 forum threads, updated frequently.
*   **Searchable Plain Text:** Content is stored in plain text for easy searching and analysis.
*   **Automated Archiving:** New threads with significant activity are backed up.
*   **Regular Updates:** Archive is updated, with a rolling window of recent discussions.
*   **Historical Archives:** Historical archives available for each year.

## How it Works

This repository automatically backs up new S1 forum threads that meet specific criteria. Threads are archived in files of approximately 1500 posts, and are easily searchable. Threads are considered active if they meet these requirements:

*   Must contain over 40 posts on the first page within 24 hours.
*   Threads are available for 3 days.
*   Files are removed after 3 days if they have not been updated.

Older content is moved to the historical archives.

## Supported Formatting

The backup process supports the following formatting elements:

*   **Bold text**
*   Links
*   Images (jpg, jpeg, png, gif, tif, webp)
*   Ratings

## Additional Resources

*   **Local Backup Tool:** For a complete local backup with images and more, consider using the [S1Downloader](https://github.com/shuangluoxss/Stage1st-downloader).
*   **COVID-19 Threads:** Backups of the COVID-19 related threads are available at [https://gitlab.com/memory-s1/virus](https://gitlab.com/memory-s1/virus).

## Historical Archives

Access archived discussions from previous years:

| 2020-2021 ([S1PlainTextArchive2021](https://github.com/TomoeMami/S1PlainTextArchive2021)) | 2022 ([S1PlainTextArchive2022](https://github.com/TomoeMami/S1PlainTextArchive2022)) | 2023 ([S1PlainTextArchive2023](https://github.com/TomoeMami/S1PlainTextArchive2023)) | 2024 ([S1PlainTextArchive2024](https://github.com/TomoeMami/S1PlainTextArchive2024)) | 2025+ ([S1PlainTextArchive2025](https://github.com/TomoeMami/S1PlainTextArchive2025)) |
|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|

## Important Notes

*   **File Size Limit:** Individual files are limited to 1MB to ensure proper rendering on GitHub. Therefore, threads are split into multiple files when they exceed this size.
*   **Update Frequency & Retention:** The archive is updated regularly.

## Changelog

*   **2024-02-15:** Updated to collect threads with over 40 replies on the first page within 24 hours. Increased caching to 14 days.
*   **2024-02-03:** Adjusted to collect threads with over 40 replies on the first page within 12 hours. Reduced caching to 7 days.