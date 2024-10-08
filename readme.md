# PDF and Web Content Chat App

This Streamlit application allows users to chat with multiple PDFs and web content using various Language Models (LLMs). It processes PDF documents and scrapes web content, creates vector embeddings, and enables users to ask questions about the processed information.

## Features

- Upload and process multiple PDF documents
- Scrape and process web content from a given URL
- Choose between different LLMs (Ollama, OpenAI, Claude via AWS Bedrock)
- Interactive chat interface to ask questions about the processed content
- Vector store creation and management using FAISS

## Requirements

- Python 3.7+
- Streamlit
- PyPDF2
- LangChain
- Hugging Face Transformers
- FAISS
- BeautifulSoup
- Requests
- Ollama (optional, for local LLM)
- OpenAI API key (optional, for OpenAI GPT)
- AWS credentials (optional, for Claude via Bedrock)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/pdf-web-chat-app.git
   cd pdf-web-chat-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables or Streamlit secrets for API keys:
   - For OpenAI: Set the `openai_api_key` in your Streamlit secrets
   - For AWS Bedrock: Configure your AWS credentials

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the provided local URL (usually `http://localhost:8501`).

3. Upload PDF documents and/or enter a URL to scrape.

4. Click the "Process" button to analyze the content.

5. Select your preferred LLM from the dropdown menu.

6. Start asking questions about the processed content in the text input field.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.