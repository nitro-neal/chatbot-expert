
# Code Block #1
!pip install langchain
!pip install InstructorEmbedding
!pip install sentence_transformers
!pip install faiss-gpu
!pip install openai
!pip install tqdm

# Code Block #2
from google.colab import drive
drive.mount('/content/drive/')

# Code Block #3
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm

data_dirpath = '/content/drive/My Drive/chatbot/input'
embedding_cache_folder = '/content/drive/My Drive/chatbot/embedding'
index_save_directory = '/content/drive/My Drive/chatbot/faiss'

loader = DirectoryLoader (path=data_dirpath, glob="*.txt", loader_cls=TextLoader)
essays = loader.load()

text_splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=100)
documents = text_splitter.split_documents(essays)
documents_with_progress = tqdm(documents, total=len(documents))

print(len(documents_with_progress))

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"}, cache_folder=embedding_cache_folder,)

vectorstore = FAISS.from_documents(documents_with_progress, embeddings)
vectorstore.save_local(index_save_directory)

# Code Block #4
import os
from langchain import PromptTemplate, OpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = 'YOUR-OPEN-AI-API-KEY'
embedding_cache_folder = '/content/drive/My Drive/chatbot/embedding'
index_save_directory = '/content/drive/My Drive/chatbot/faiss'

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"}, cache_folder=embedding_cache_folder,)


vectorstore = FAISS.load_local(index_save_directory, embeddings)

llm = OpenAI(temperature=0)


prompt_template = PromptTemplate(
  input_variables=['context', 'question'],
  template="Use the following pieces of context to answer the question at the end. If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.\n\n{context}In\nQuestion: {question} \nHelpful Answer:"
)


qa_chain = RetrievalQA.from_llm(
    llm=llm,
    prompt=prompt_template,
    retriever=vectorstore.as_retriever()
)

query = "Who wrote the declaration of independence?"
print(qa_chain(query, return_only_outputs=True))
