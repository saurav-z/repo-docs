# yt-fts: Unlock YouTube's Secrets with Powerful Search & Summarization 

**yt-fts empowers you to search and summarize YouTube videos using their subtitles with full-text and semantic search, giving you instant access to the information you need.** [View the original repository on GitHub](https://github.com/NotJoeMartinez/yt-fts).

<img src="https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14" alt="yt-fts demo" width="500"/>

## Key Features

*   **Full-Text Search:** Quickly find specific keywords or phrases within YouTube video transcripts using a simple command-line interface.
*   **Semantic Search:** Leverage OpenAI, Gemini, and ChromaDB to understand the meaning of your search queries and find relevant videos.
*   **Video Summarization:** Generate concise summaries of YouTube videos with time-stamped links, saving you time and effort.
*   **LLM Chat Bot:** Have interactive conversations with LLMs powered by the semantic search results of your initial prompt as context.
*   **Channel and Playlist Support:** Download and search subtitles from entire YouTube channels or playlists.
*   **Flexible Export Options:** Export search results and transcripts in various formats (CSV, TXT, VTT).
*   **Error Diagnostics:** Troubleshoot common download issues with the `diagnose` command.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download` - Download Subtitles

Download subtitles for a YouTube channel or playlist.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download all videos from a playlist
*   `-l, --language`: Subtitle language (default: en)
*   `-j, --jobs`: Parallel download jobs (default: 8)
*   `--cookies-from-browser`: Browser for cookie retrieval

### `diagnose` - Diagnose Download Issues

Test and troubleshoot common YouTube download problems.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test
*   `--cookies-from-browser`: Browser for cookie retrieval
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8)

### `list` - List Saved Content

View your saved channels, videos, and transcripts.

```bash
yt-fts list
yt-fts list --channel "3Blue1Brown"
yt-fts list --transcript "dQw4w9WgXcQ"
```

**Options:**

*   `-t, --transcript`: Show transcript for a video
*   `-c, --channel`: Show videos for a channel
*   `-l, --library`: Show a list of channels in your library

### `update` - Update Subtitles

Update subtitles for all or specific channels.

```bash
yt-fts update
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Channel to update
*   `-l, --language`: Subtitle language (default: en)
*   `-j, --jobs`: Parallel download jobs (default: 8)
*   `--cookies-from-browser`: Browser for cookie retrieval

### `delete` - Delete Channel

Delete a channel and all associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel to delete (required)

### `export` - Export Transcripts

Export transcripts in TXT or VTT format.

```bash
yt-fts export --channel "3Blue1Brown" --format txt
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel to export (required)
*   `-f, --format`: Export format (txt, vtt; default: txt)

### `search` - Full Text Search

Perform full-text searches within your saved transcripts.

```bash
yt-fts search "[search query]" --channel "[channel name or id]"
yt-fts search "[search query]" --video-id "[video id]"
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts search "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Channel to search in
*   `-v, --video-id`: Video ID to search in
*   `-l, --limit`: Maximum results (default: 10)
*   `-e, --export`: Export results to CSV

**Advanced Search Syntax:** Use SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for more advanced searches, including AND/OR operators and wildcards.

```bash
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### Semantic Search and RAG

Enable semantic search for more powerful searching, summarization and an LLM chat bot. To use semantic search, you must set your OpenAI or Gemini API Key as an environment variable, or use the `--api-key` flag.

### `embeddings` - Generate Embeddings

Generate embeddings for a channel to enable semantic search.

```bash
# Requires OpenAI or Gemini API key
# export OPENAI_API_KEY="[yourOpenAIKey]"
yt-fts embeddings --channel "3Blue1Brown"
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel to generate embeddings for
*   `--api-key`: API key (if not provided, uses environment variable)
*   `-i, --interval`: Transcript chunk interval in seconds (default: 30)

### `vsearch` - Semantic Search

Search using semantic understanding (requires `embeddings` command).

```bash
yt-fts vsearch "[search query]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --video-id "[video id]"
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Channel to search in
*   `-v, --video-id`: Video ID to search in
*   `-l, --limit`: Maximum results (default: 10)
*   `-e, --export`: Export results to CSV
*   `--api-key`: API key (if not provided, uses environment variable)

### `llm` - Chat Bot

Interact with a LLM using the semantic search results as context.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel to use (required)
*   `--api-key`: API key (if not provided, uses environment variable)

### `summarize` - Summarize Video

Get concise summaries of YouTube videos.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
yt-fts summarize "9-Jl0dxWQs8"
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use (e.g., gpt-3.5-turbo)
*   `--api-key`: API key (if not provided, uses environment variable)

### `config` - Show Configuration

Display the current configuration, including database and Chroma paths.

```bash
yt-fts config
```

## How To

**Export Search Results:**

Export the results of your `search` and `vsearch` commands to a CSV file using the `--export` flag.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

Delete a channel using the `delete` command.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

Update the full text search data for a channel using the `update` command (this command will not update semantic search embeddings).

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export All of a Channel's Transcript:**

Export a channel's transcripts to a text or VTT file:

```bash
yt-fts export --channel "[id/name]" --format "[vtt/txt]"