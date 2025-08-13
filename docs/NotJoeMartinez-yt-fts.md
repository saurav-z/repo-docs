# yt-fts: Unleash the Power of YouTube Search with Full Text and Semantic Capabilities

Tired of endless scrolling through YouTube?  **yt-fts** is your command-line solution for in-depth YouTube video exploration, offering full-text and semantic search capabilities powered by subtitles.  [Explore the yt-fts repository](https://github.com/NotJoeMartinez/yt-fts).

![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)

**Key Features:**

*   **Full Text Search:**  Quickly find specific keywords and phrases within YouTube video transcripts using advanced search syntax.
*   **Semantic Search (powered by OpenAI/Gemini/ChromaDB):**  Discover videos based on meaning and context, even with imperfect search terms.
*   **Channel and Playlist Downloads:** Easily download subtitles for entire channels or playlists for offline access and analysis.
*   **LLM-Powered Chat Bot:** Engage in conversational exploration of video content, leveraging semantic search for context.
*   **Video Summarization:** Quickly get the gist of any video with time-stamped summaries.
*   **Flexible Export Options:** Export search results and transcripts in various formats (CSV, TXT, VTT).
*   **Database Driven:** Utilizes an SQLite database for efficient storage and fast retrieval of your YouTube video transcripts.
*   **Error Diagnostics:** Troubleshoot download issues with the built-in diagnose feature.

## Getting Started

### Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download` - Download Subtitles

Download subtitles for a channel or playlist.

*   `yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"` (Download channel with 5 parallel jobs)
*   `yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"` (Download using browser cookies)
*   `yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"` (Download playlist)

**Options:**

*   `-p, --playlist`: Download from a playlist.
*   `-l, --language`: Subtitle language (default: en).
*   `-j, --jobs`: Number of parallel download jobs (default: 8, recommended: 4-16).
*   `--cookies-from-browser`: Use cookies from your browser (chrome, firefox, etc.)

### `diagnose` - Diagnose Download Issues

Test your connection and resolve 403 errors.

*   `yt-fts diagnose` (Run diagnostic tests)
*   `yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox` (Test with a specific URL)

**Options:**

*   `-u, --test-url`: URL to test.
*   `--cookies-from-browser`: Browser to extract cookies from.
*   `-j, --jobs`: Number of parallel jobs for testing.

### `list` - List Saved Content

List saved channels, videos, and transcripts.

*   `yt-fts list` (List all channels)
*   `yt-fts list --channel "3Blue1Brown"` (List videos for a channel)
*   `yt-fts list --transcript "dQw4w9WgXcQ"` (Show transcript for a video)

**Options:**

*   `-t, --transcript`: Show transcript for a video.
*   `-c, --channel`: Show videos for a channel.
*   `-l, --library`: Show list of channels in library.

### `update` - Update Subtitles

Update subtitles for all or specific channels.

*   `yt-fts update` (Update all channels)
*   `yt-fts update --channel "3Blue1Brown" --jobs 5` (Update a channel)

**Options:**

*   `-c, --channel`: The name or id of the channel to update.
*   `-l, --language`: Language of the subtitles to download (default: en).
*   `-j, --jobs`: Number of parallel download jobs (default: 8).
*   `--cookies-from-browser`: Browser to extract cookies from.

### `delete` - Delete Channels

Delete a channel and all its data.

*   `yt-fts delete --channel "3Blue1Brown"` (Delete a channel, requires confirmation).

**Options:**

*   `-c, --channel`: The name or id of the channel to delete (required).

### `export` - Export Transcripts

Export transcripts for a channel.

*   `yt-fts export --channel "3Blue1Brown" --format txt` (Export to TXT)
*   `yt-fts export --channel "3Blue1Brown" --format vtt` (Export to VTT)

**Options:**

*   `-c, --channel`: The name or id of the channel to export.
*   `-f, --format`: Export format (txt, vtt, default: txt).

### `search` - Full Text Search

Search for keywords in saved transcripts.

*   `yt-fts search "[search query]"` (Search all channels)
*   `yt-fts search "[search query]" --channel "[channel name or id]"` (Search a specific channel)
*   `yt-fts search "[search query]" --video-id "[video id]"` (Search a specific video)
*   `yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"` (Limit Results)
*   `yt-fts search "[search query]" --export --channel "[channel name or id]"` (Export Results)

**Options:**

*   `-c, --channel`: The name or id of the channel to search in.
*   `-v, --video-id`: The id of the video to search in.
*   `-l, --limit`: Number of results to return (default: 10).
*   `-e, --export`: Export search results to a CSV file.

**Advanced Search Syntax:** Use SQLite's Enhanced Query Syntax for more advanced searches including `AND`, `OR` and wildcards.

*   `yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"`
*   `yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"`
*   `yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"`

### Semantic Search and RAG

Requires `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable or `--api-key` flag.

### `embeddings` - Generate Semantic Search Embeddings

Generate embeddings for semantic search.

*   `yt-fts embeddings --channel "3Blue1Brown"` (Generate embeddings for a channel)
*   `yt-fts embeddings --interval 60 --channel "3Blue1Brown"` (Set the time interval)

**Options:**

*   `-c, --channel`: The name or id of the channel.
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY).
*   `-i, --interval`: Interval in seconds to split transcripts (default: 30).

### `vsearch` - Semantic (Vector) Search

Search using semantic similarity.

*   `yt-fts vsearch "[search query]" --channel "[channel name or id]"` (Search by channel name)
*   `yt-fts vsearch "[search query]" --video-id "[video id]"` (Search in a specific video)
*   `yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"` (Limit Results)
*   `yt-fts vsearch "[search query]" --export --channel "[channel name or id]"` (Export Results)

**Options:**

*   `-c, --channel`: The name or id of the channel to search in.
*   `-v, --video-id`: The id of the video to search in.
*   `-l, --limit`: Number of results to return (default: 10).
*   `-e, --export`: Export search results to a CSV file.
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY).

### `llm` - LLM Chat Bot

Chat with the data using a Large Language Model.  Requires semantic search to be enabled.

*   `yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"` (Ask the Chatbot a question)

**Options:**

*   `-c, --channel`: The name or id of the channel to use (required).
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY).

### `summarize` - Summarize Video Transcripts

Summarize video transcripts.

*   `yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"` (Summarize a video by URL)
*   `yt-fts summarize "9-Jl0dxWQs8"` (Summarize by video ID)
*   `yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"` (Use a different model)

**Options:**

*   `--model, -m`: Model to use for summarization.
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY).

### `config` - Show Configuration

View configuration settings, including database and chroma paths.

*   `yt-fts config`

## How To

**Export Search Results:**

Export results to CSV using the `--export` flag with either the `search` or `vsearch` commands.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

Delete a channel and all of its data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

Update existing transcripts.

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export all of a channel's transcript:**

This command will create a directory in current working directory with the YouTube
channel id of the specified channel.

```bash
# Export to vtt
yt-fts export --channel "[id/name]" --format "[vtt/txt]"