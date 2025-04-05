# Medical Assistant Chatbot

A sophisticated medical chatbot that provides accurate medical information using advanced natural language processing and retrieval-augmented generation techniques.

## Features

- Medical question answering with context-aware responses
- Professional and accurate medical information retrieval
- Error handling and validation for reliable operation
- Rate limiting to prevent abuse
- Vector-based semantic search for accurate information retrieval
- Source document citation for transparency

## Technologies Used

- Python 3.x
- LangChain
- HuggingFace Transformers
- FAISS Vector Store
- Sentence Transformers
- Mistral AI Model

## Requirements

- Python 3.x
- HuggingFace API Token
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone 
cd 
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file and add your HuggingFace API token:
```bash
HF_TOKEN=your_huggingface_token_here
```

## Usage

Run the main application:
```bash
python main.py
```

The chatbot will initialize and you can start asking medical-related questions.

## Project Structure

- `main.py` - Main application entry point
- `connect_memory_with_llm.py` - LLM connection and chain setup
- `create_memory_for_llm.py` - Vector store creation and management
- `csv_generate.py` - Data processing utilities
- `vectorstore/` - FAISS vector database storage
- `model_cache/` - Local model cache for better performance
- `data/` - Source medical data and documents

## Best Practices

- Always verify medical information with healthcare professionals
- The chatbot is for informational purposes only
- Not a replacement for professional medical advice
- Consult healthcare providers for medical decisions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Safety Note

This chatbot is designed for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.