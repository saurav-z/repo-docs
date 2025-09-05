# S1 Forum Plain Text Backup - Preserving Stage1st Forum History

**Preserve and access your favorite Stage1st (S1) forum threads with this plain text backup, offering a searchable and archivable history.**

[View the original repository on GitHub](https://github.com/TomoeMami/S1PlainTextBackup)

## Key Features

*   **Comprehensive Backup:** Stores plain text backups of active S1 forum threads.
*   **Regular Updates:** Focuses on new threads with activity, ensuring a current snapshot of the forum.
*   **Searchable Content:** Plain text format allows for easy searching and content discovery.
*   **Archive Structure:** Threads are organized into files with approximately 1500 posts per file, making navigation and retrieval easier.
*   **Format Support:** Preserves basic formatting to enhance readability:
    *   Bold text
    *   Hyperlinks
    *   Image support (jpg, jpeg, png, gif, tif, webp)
    *   Ratings

## How It Works

This repository automatically backs up new threads from the Stage1st forum. Specifically, it collects threads that meet the following criteria:

*   Threads must have more than 40 replies (1 page) within the first 24 hours.
*   Backups are stored for 14 days.
*   After 3 days without replies, the backup will be moved to the appropriate historical archive.

## Historical Archives

Access historical backups by year:

| [2020-2021](https://github.com/TomoeMami/S1PlainTextArchive2021) |
| [2022](https://github.com/TomoeMami/S1PlainTextArchive2022) | [2023](https://github.com/TomoeMami/S1PlainTextArchive2023) | [2024](https://github.com/TomoeMami/S1PlainTextArchive2024) |
| [2025-Present](https://github.com/TomoeMami/S1PlainTextArchive2025) |

## Important Notes

*   **File Size:** To ensure proper rendering on GitHub, individual files are limited to 1MB and are split into multiple files with a maximum of 50 pages.
*   **Local Downloads:** For a complete local backup with images, consider using the [S1Downloader](https://github.com/shuangluoxss/Stage1st-downloader) tool.
*   **COVID-19 Thread Backups:** Backups for the COVID-19-related threads (three and four) are sourced from [https://gitlab.com/memory-s1/virus](https://gitlab.com/memory-s1/virus).

## Changelog

*   **February 15, 2024:** Adjusted collection criteria to include threads with over 40 replies (1 page) within 24 hours. Increased the cache to 14 days.
*   **February 3, 2024:** Modified collection to include threads with over 40 replies (1 page) within 12 hours. Reduced the cache to 7 days.