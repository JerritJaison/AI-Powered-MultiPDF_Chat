# ChatPDF - AI-Powered Multi-PDF Chat System

## Overview
ChatPDF is an AI-powered chatbot that allows users to upload multiple PDFs and interact with their content using natural language queries. It leverages **LangChain, Streamlit, Google Gemini API, and FAISS** for efficient document retrieval and question answering.

## Features
-  **Multi-PDF Upload**: Supports multiple PDF documents for analysis.
-  **AI-Powered Q&A**: Uses Google Gemini to generate responses.
-  **Efficient Text Retrieval**: Implements FAISS for fast document search.
-  **Interactive UI**: Built with Streamlit for a smooth user experience.
-  **Conversational Memory**: Maintains chat history for better context.

##  Tech Stack
- **Python**
- **Streamlit** (UI Framework)
- **LangChain** (AI Processing)
- **Google Gemini API** (LLM Model)
- **FAISS** (Vector Search)
- **PyPDF2** (PDF Processing)

## Installation
### **1️ Clone the Repository**
```bash
git clone https://github.com/JerritJaison/chatPDF.git
cd chatPDF
```

### **2️ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️ Set Up API Keys**
Create a `.env` file and add your API key:
```
GEMINI_API_KEY=your_google_gemini_api_key
```

### **4️ Run the Application**
```bash
streamlit run app.py
```

##  Usage
1. Upload multiple PDFs using the Streamlit sidebar.
2. Ask questions related to the uploaded documents.
3. The chatbot will retrieve relevant text and generate responses.
4. View chat history for context-aware responses.

##  Future Improvements
-  Add support for more file formats (DOCX, TXT)
-  Improve LLM fine-tuning for better accuracy
-  Optimize vector search for large-scale documents

##  Contributing
Feel free to fork this repository, submit pull requests, or report issues!

## License
This project is licensed under the MIT License.

---
**🔗 Connect with Me:** [GitHub](https://github.com/JerritJaison) | [Email](jeritjaison995@gmail.com)

