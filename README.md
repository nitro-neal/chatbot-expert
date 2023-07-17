# Chatbot Expert

This is a Python script to be run in Google Colab that will serve as an expert on any text files you upload into your Google Drive folder at googledrive/chatbot/input. The chatbot utilizes OpenAI's language model and several natural language processing libraries to provide useful insights based on the input text files.

The code is divided into several blocks which are intended to be copy-pasted into separate Google Colab code blocks for execution.
Dependencies

The following packages are required and are installed in the first code block:

    *langchain
    *InstructorEmbedding
    *sentence_transformers
    *faiss-gpu
    *openai
    *tqdm

# Structure

Code Block #1: Installs the necessary dependencies.

Code Block #2: Mounts your Google Drive to the Colab notebook.

Code Block #3: This block is dedicated to loading text files from the specified directory (/content/drive/My Drive/chatbot/input), splitting the content, creating the embeddings using the hkunlp/instructor-large model, and saving these embeddings into a FAISS vector store.

Code Block #4: This section retrieves the embeddings from the FAISS vector store, creates a Language Model (LM) using the OpenAI API, and sets up a template for the prompt. The chatbot expert retrieves relevant information based on this prompt, providing an answer to the user's query. The OPENAI_API_KEY should be replaced with your actual OpenAI API key.
Usage

To utilize this chatbot expert, follow the instructions below:

    * Copy each of the code blocks individually into separate Google Colab code blocks.

    * Replace 'YOUR-OPEN-AI-API-KEY' in Code Block #4 with your OpenAI API key.

    * Upload the text files you want the chatbot to learn from to your Google Drive, specifically in the googledrive/chatbot/input directory.

    * Run each of the code blocks in order.

    * Query the chatbot by replacing "Who wrote the declaration of independence?" in the last line of Code Block #4 with your own question.

The chatbot will then use the embeddings generated from the text files and OpenAI's Language Model to answer the questions based on the knowledge it has gained from the text files.

Note: Ensure you have sufficient Google Drive space available for storing the input text files, embedding, and FAISS vector store. This code requires a GPU; ensure that your Google Colab environment is set to use GPU acceleration.

# Limitations

The effectiveness of the chatbot expert is reliant on the quality and relevance of the text files it is trained on. Poorly formatted text or text that does not cover the relevant subjects can negatively impact the performance of the chatbot. It's important to curate and format your text files appropriately for the best results.