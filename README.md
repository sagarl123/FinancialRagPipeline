
# Financial RAG Pipeline

A sophisticated Retrieval-Augmented Generation (RAG) system designed to analyze and extract insights from financial PDF reports using LangChain, Ollama, and multimodal processing capabilities.

## Features

- **PDF Content Extraction**: Processes complex financial PDFs with text, tables, and images

- **Multimodal Analysis**: Handles text, tabular data, and image content from financial documents

- **Vector Storage**: Uses Chroma for efficient similarity search and retrieval

- **Multi-Model Support**: Integrates both local (Ollama) and cloud-based (OpenAI) language models

- **Financial Expertise**: Specialized prompts for financial report analysis and auditing

## Architecture

The pipeline consists of several key components:

- **Document Processing**: Extracts and chunks PDF content using unstructured library

- **Summary Generation**: Creates concise summaries for text, tables, and images

- **Vector Storage**: Stores embeddings in Chroma with MultiVectorRetriever for efficient retrieval

- **Query Interface**: Natural language querying of financial documents

## Prerequisites

### System Requirements

- Python 3.8+

- Ollama installed locally

- Sufficient disk space for vector database storage

### Required Models

Install the following Ollama models:

```bash

ollama pull llama3.2:3b

ollama pull gemma3:4b

```

## Installation

1. **Clone the repository**

```bash

git clone <your-repo-url>

cd financial-rag-pipeline

```

2. **Install dependencies**

```bash

pip install -r requirements.txt

```

3. **Set up environment variables**

Create a `.env` file in the root directory:

```env

OPENAI_API_KEY=your_openai_api_key_here

```

## Dependencies

Create a `requirements.txt` file with these dependencies:

```txt

langchain-ollama

langchain-chroma

langchain-core

langchain-openai

langchain

unstructured[pdf]

python-dotenv

chromadb

uuid

logging

```

## Usage

### Basic Usage

1. **Place your PDF file**

```bash

mkdir pdf

# Copy your financial report PDF to the pdf/ directory

```

2. **Update the PDF path**

Modify the `pdf_path` variable in the `main()` function:

```python

pdf_path = 'pdf/your_financial_report.pdf'

```

3. **Run the pipeline**

```bash

python financial_rag_pipeline.py

```

### Custom Queries

Modify the questions list in the `main()` function to ask specific questions:

```python

questions = [

"What is the total revenue for this quarter?",

"What are the operating expenses?",

"Summarize the cash flow statement",

"What are the key risk factors mentioned?"

]

```

### API Usage

```python

from financial_rag_pipeline import FinancialRAGPipeline

# Initialize the pipeline

pipeline = FinancialRAGPipeline('path/to/your/financial_report.pdf')

# Run the processing pipeline

pipeline.run_pipelines()

# Query the processed document

answer = pipeline.query("What are the main financial highlights?")

print(answer)

```

## Configuration

### Model Configuration

The system uses multiple models for different tasks:

- **Text Processing**: `llama3.2:3b` (Ollama)

- **Image Analysis**: `gemma3:4b` (Ollama)

- **Backup/Fallback**: `gpt-4-mini` (OpenAI)

### Storage Configuration

- **Vector Store**: Chroma database stored in `./chroma_langchain_db`

- **Document Store**: Local file store in `./local_docstore`

### Processing Parameters

Key parameters you can adjust:

```python

# PDF processing settings

max_characters = 7000

combine_text_under_n_chars = 2000

new_after_n_chars = 6000

# Retrieval settings

similarity_search_k = 3

# Model settings

temperature = 0.5 # for local model

max_retries = 2 # for OpenAI model

```

## File Structure

```

financial-rag-pipeline/

├── financial_rag_pipeline.py # Main pipeline code

├── requirements.txt # Python dependencies

├── .env # Environment variables

├── .gitignore # Git ignore file

├── README.md # This file

├── pdf/ # Directory for PDF files

│  └── your_report.pdf

├──  chroma_langchain_db/ # Vector database (created automatically)

└──  local_docstore/ # Document storage (created automatically)

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangChain](https://python.langchain.com/) framework

- Uses [Ollama](https://ollama.ai/) for local language models

- PDF processing powered by [Unstructured](https://unstructured.io/)

- Vector storage provided by [Chroma](https://www.trychroma.com/)

## Support

For questions and support, please open an issue on GitHub or contact [your-email@example.com].

---

**Note**: This system is designed for educational and research purposes. Always verify financial analysis results with qualified professionals before making investment decisions.
