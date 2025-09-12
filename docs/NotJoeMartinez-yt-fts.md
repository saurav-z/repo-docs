# yt-fts: Unleash the Power of YouTube Video Search

Tired of endless scrolling? **yt-fts** is a powerful command-line tool that lets you full-text search YouTube video transcripts, turning the vast library of YouTube content into an easily searchable database.  [Check out the original repo](https://github.com/NotJoeMartinez/yt-fts).

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

**Key Features:**

*   **Full-Text Search:** Quickly find specific keywords or phrases within YouTube video transcripts using the command line.
*   **Semantic Search:** Leverage the power of AI with OpenAI or Gemini embeddings to understand the context of your search queries.
*   **Time-Stamped Results:** Get precise timestamps for each search result, linking directly to the relevant video segments.
*   **Flexible Data Management:** Download, update, list, and delete YouTube channel data with ease.
*   **Advanced Search Syntax:** Utilize SQLite's Enhanced Query Syntax for powerful search capabilities.
*   **LLM Integration:** Interact with a chatbot using search results as context.
*   **Video Summarization:** Generate concise summaries of YouTube videos directly from their transcripts.

## Getting Started

### Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### Download

Download video transcripts for a channel or playlist and store them in the local database.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
```

**Options:**

*   `-p, --playlist`: Download all videos from a playlist.
*   `-l, --language`: Specify the subtitle language (default: `en`).
*   `-j, --jobs`: Set the number of parallel download jobs (default: `8`, recommended: `4-16`).
*   `--cookies-from-browser`: Use cookies from your browser (e.g., `firefox`) to handle potential sign-in requests.

### Diagnose

Diagnose and troubleshoot potential download issues.

```bash
yt-fts diagnose
```

**Options:**

*   `-u, --test-url`: Test a specific video URL.
*   `--cookies-from-browser`: Use cookies from your browser.
*   `-j, --jobs`: Set the number of parallel download jobs.

### List

List stored channels, videos, and transcripts.

```bash
yt-fts list
```

**Options:**

*   `-t, --transcript`: Show the transcript for a specific video.
*   `-c, --channel`: Show a list of videos for a specific channel.
*   `-l, --library`: Show a list of all channels in the library.

### Update

Update the transcripts for existing channels in your library.

```bash
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: The name or ID of the channel to update.
*   `-l, --language`: Specify the subtitle language (default: `en`).
*   `-j, --jobs`: Set the number of parallel download jobs.
*   `--cookies-from-browser`: Use cookies from your browser.

### Delete

Delete a channel and all its associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or ID of the channel to delete (required).

### Export

Export transcripts to local files.

```bash
yt-fts export --channel "3Blue1Brown" --format txt
```

**Options:**

*   `-c, --channel`: The name or ID of the channel to export transcripts for (required).
*   `-f, --format`: The format to export transcripts to (txt, vtt; default: txt).

### Search

Full-text search within saved transcripts.

```bash
yt-fts search "keyword or phrase" --channel "channel name or id"
```

**Options:**

*   `-c, --channel`: Search within a specific channel.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`: Limit the number of results.
*   `-e, --export`: Export results to a CSV file.

**Advanced Search Syntax:**

Use [SQLite's Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for advanced search options, including AND, OR, and wildcard searches.

```bash
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### Semantic Search and RAG

Enable semantic search for more intelligent results by leveraging AI embeddings.

1.  **Set your API Key:** Set either `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variables, or use the `--api-key` flag with the commands.

```bash
# export OPENAI_API_KEY="[yourOpenAIKey]"
yt-fts embeddings --channel "3Blue1Brown"
```

2.  **Run the Embeddings command**

```bash
yt-fts embeddings --channel "3Blue1Brown"
```

*   `-c, --channel`: The name or ID of the channel to generate embeddings for.
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable).
*   `-i, --interval`: Interval in seconds to split the transcripts into chunks (default: 30).

### Vsearch (Semantic Search)

Perform semantic (vector) searches based on your embeddings.

```bash
yt-fts vsearch "relevant topic" --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Search within a specific channel.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`: Limit the number of results.
*   `-e, --export`: Export results to a CSV file.
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable).

### LLM (Chat Bot)

Interact with a chatbot, using the semantic search results of your initial prompt as context to answer questions.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: The name or ID of the channel to use (required).
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable).

### Summarize

Summarize YouTube video transcripts.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use for summarization.
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable).

### Config

Display configuration settings, including database and chroma paths.

```bash
yt-fts config
```

## Example Usage

*   **Export search results:**

    ```bash
    yt-fts search "life in the big city" --export
    yt-fts vsearch "existing in large metropolaten center" --export
    ```

*   **Delete a channel:**

    ```bash
    yt-fts delete --channel "3Blue1Brown"
    ```

*   **Update a channel:**

    ```bash
    yt-fts update --channel "3Blue1Brown"
    ```

*   **Export a channel's transcript:**

    ```bash
    yt-fts export --channel "[id/name]" --format "[vtt/txt]"
    ```