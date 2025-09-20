# yt-fts: Unleash the Power of YouTube Full Text Search

**Unlock the hidden knowledge within YouTube by searching video transcripts with ease using `yt-fts`!**  ([Original Repo](https://github.com/NotJoeMartinez/yt-fts))

This powerful command-line program utilizes `yt-dlp` to scrape subtitles from YouTube channels, store them in a searchable SQLite database, and offers advanced features like semantic search and AI-powered summaries.

<img src="https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14" alt="yt-fts demo" width="600"/>

## Key Features

*   **Full-Text Search:** Quickly find videos containing specific keywords or phrases across multiple channels.
*   **Semantic Search:** Leverage OpenAI or Gemini embeddings to search based on the meaning of your query, discovering relevant videos even with different wording.
*   **AI-Powered Summaries:** Generate concise summaries of video transcripts, complete with timestamps for easy navigation.
*   **LLM/RAG Chat Bot:** Have a conversation with your data. Ask questions about the content of the video and get answers!
*   **Flexible Data Management:** Download, update, list, delete, and export channel data with simple commands.
*   **Advanced Search Syntax:** Utilize SQLite's enhanced query syntax for complex searches, including AND/OR operators and wildcards.
*   **Easy Installation:** Install with a simple `pip install yt-fts` command.

## Installation

```bash
pip install yt-fts
```

## Commands

### `download`
Download subtitles for a channel or playlist.

**Options:**
*   `-p, --playlist`: Download all videos from a playlist
*   `-l, --language`: Language of the subtitles to download (default: en)
*   `-j, --jobs`: Number of parallel download jobs (default: 8, recommended: 4-16)
*   `--cookies-from-browser`: Browser to extract cookies from (chrome, firefox, etc.)

```bash
# Download channel
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"

# Download playlist
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

### `diagnose`
Diagnose 403 errors and other download issues.

**Options:**
*   `-u, --test-url`: URL to test with (default: https://www.youtube.com/watch?v=dQw4w9WgXcQ)
*   `--cookies-from-browser`: Browser to extract cookies from
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8)

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

### `list`
List saved channels, videos, and transcripts.

**Options:**
*   `-t, --transcript`: Show transcript for a video
*   `-c, --channel`: Show list of videos for a channel
*   `-l, --library`: Show list of channels in library

```bash
# List all channels
yt-fts list

# List videos for a specific channel
yt-fts list --channel "3Blue1Brown"

# Show transcript for a specific video
yt-fts list --transcript "dQw4w9WgXcQ"

# Show library (same as default)
yt-fts list --library
```

### `update`
Update subtitles for all channels in the library or a specific channel.

**Options:**
*   `-c, --channel`: The name or id of the channel to update
*   `-l, --language`: Language of the subtitles to download (default: en)
*   `-j, --jobs`: Number of parallel download jobs (default: 8)
*   `--cookies-from-browser`: Browser to extract cookies from

```bash
# Update all channels
yt-fts update

# Update specific channel
yt-fts update --channel "3Blue1Brown" --jobs 5
```

### `delete`
Delete a channel and all its data.

**Options:**
*   `-c, --channel`: The name or id of the channel to delete (required)

```bash
yt-fts delete --channel "3Blue1Brown"
```

### `export`
Export transcripts for a channel.

**Options:**
*   `-c, --channel`: The name or id of the channel to export transcripts for (required)
*   `-f, --format`: The format to export transcripts to. Supported formats: txt, vtt (default: txt)

```bash
# Export to txt format (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to vtt format
yt-fts export --channel "3Blue1Brown" --format vtt
```

### `search` (Full Text Search)
Full text search for a string in saved channels.

**Options:**
*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file

```bash
# search in all channels
yt-fts search "[search query]"

# search in channel
yt-fts search "[search query]" --channel "[channel name or id]"

# search in specific video
yt-fts search "[search query]" --video-id "[video id]"

# limit results
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# export results to csv
yt-fts search "[search query]" --export --channel "[channel name or id]"
```

**Advanced Search Syntax:**

The search string supports sqlite [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries).
which includes things like [prefix queries](https://www.sqlite.org/fts3.html#termprefix) which you can use to match parts of a word.

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"

# OR SEARCH
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"

# wild cards
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### `embeddings` (Semantic Search)
Fetches embeddings for specified channel

**Options:**
*   `-c, --channel`: The name or id of the channel to generate embeddings for
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)
*   `-i, --interval`: Interval in seconds to split the transcripts into chunks (default: 30)

```bash
# make sure API key is set
# export OPENAI_API_KEY="[yourOpenAIKey]"
# or
# export GEMINI_API_KEY="[yourGeminiKey]"

yt-fts embeddings --channel "3Blue1Brown"

# specify time interval in seconds to split text by default is 30
# the larger the interval the more accurate the llm response
# but semantic search will have more text for you to read.
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

### `vsearch` (Semantic Search)
`vsearch` is for "Vector search". This requires that you enable semantic
search for a channel with `embeddings`. It has the same options as
`search` but output will be sorted by similarity to the search string and
the default return limit is 10.

**Options:**
*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

```bash
# search by channel name
yt-fts vsearch "[search query]" --channel "[channel name or id]"

# search in specific video
yt-fts vsearch "[search query]" --video-id "[video id]"

# limit results
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# export results to csv
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
```

### `llm` (Chat Bot)
Starts interactive chat session with a model using
the semantic search results of your initial prompt as the context
to answer questions. If it can't answer your question, it has a
mechanism to update the context by running targeted query based
off the conversation. The channel must have semantic search enabled.

**Options:**
*   `-c, --channel`: The name or id of the channel to use (required)
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

### `summarize`
Summarizes a YouTube video transcript, providing time stamped URLS.
Requires a valid YouTube video URL or video ID as argument. If the
trancript is not in the database it will try to scrape it.

**Options:**
*   `--model, -m`: Model to use in summary
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
# or
yt-fts summarize "9-Jl0dxWQs8"

# Use different model
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

### `config`
Show config settings including database and chroma paths.

```bash
yt-fts config
```

## Getting Started

### Export search results:

For both the `search` and `vsearch` commands you can export the results to a csv file with
the `--export` flag. and it will save the results to a csv file in the current directory.
```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

### Delete a channel:
You can delete a channel with the `delete` command.

```bash
yt-fts delete --channel "3Blue1Brown"
```

### Update a channel:
The update command currently only works for full text search and will not update the
semantic search embeddings.

```bash
yt-fts update --channel "3Blue1Brown"
```

### Export all of a channel's transcript:

This command will create a directory in current working directory with the YouTube
channel id of the specified channel.

```bash
# Export to vtt
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```