import os
import warnings
import glob
import pickle
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "dburl-old")

source_directory = os.environ.get('SOURCE_DIRECTORY', 'single_downloaded_html_pages')

chunk_size = 2048
chunk_overlap = 200

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print("Doc :",documents)

    for d in documents:
        lines = d.page_content.split('\n')  # Split the content into lines
        processed_lines = []
        for line in lines:
            processed_line = " ".join(line.split())
            processed_lines.append(processed_line)
        d.page_content = "\n".join(processed_lines)  # Join the processed lines back together with newlines

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    return texts

# Setup the LLM model
llm = Ollama(
    model="gemma",
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

prompt_template = PromptTemplate.from_template(
    "Preparing a supervised dataset from the question and answers. Generate 5 detailed and diverse questions from the following text. Each question should explore a different aspect or detail of the text. Provide comprehensive answers that include specific references to the text. Format the response as follows: Separate the 'Questions' labeled as 'questions' and 'Answers' labeled as 'answers' in list format with labeled numbering separately. Text: {paragraph}"
)

def dataset_document(document):
    response = (prompt_template | llm).invoke({"paragraph": document.page_content})
    questions = []
    answers = []
    collecting_questions = False
    collecting_answers = False

    for line in response.split('\n'):
        line = line.strip()  # Clean up whitespace

        if line.startswith('**Questions:**'):
            collecting_questions = True
            collecting_answers = False
            continue
        elif line.startswith('**Answers:**'):
            collecting_answers = True
            collecting_questions = False
            continue
        
        if collecting_questions and line:
            if line[0].isdigit():  # Check if line starts with a digit, indicating a question number
                question = line.split('.', 1)[1].strip()
                questions.append(question)
        elif collecting_answers and line:
            if line[0].isdigit():  # Check if line starts with a digit, indicating an answer number
                answer = line.split('.', 1)[1].strip()
                answers.append(answer)

    # Check for length mismatch and adjust if necessary
    min_length = min(len(questions), len(answers))
    questions = questions[:min_length]
    answers = answers[:min_length]

    return questions, answers

def dataset_documents(documents):
    all_questions = []
    all_answers = []
    for document in documents:
        questions, answers = dataset_document(document)
        all_questions.extend(questions)
        all_answers.extend(answers)
    
    if len(all_questions) != len(all_answers):
        raise ValueError("Mismatch in the number of questions and answers generated.")
    return pd.DataFrame({
        'Question': all_questions,
        'Answer': all_answers
    })

def load_texts():
    with open('stored_texts.pkl', 'rb') as file:
        texts = pickle.load(file)
    return texts

# Create vector database
def create_vector_database():
    
    embeddings = HuggingFaceEmbeddings(model_name="Salesforce/SFR-Embedding-Mistral")
    #embeddings = HuggingFaceEmbeddings(model_name="Salesforce/SFR-Embedding-Mistral")
    vector_database = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )

    collection = vector_database.get()
    texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
    
    with open('stored_texts.pkl', 'wb') as file:
        pickle.dump(texts, file)
    
    # df = dataset_documents(texts)

    # csv_file_path = 'output_questions_answers.csv'
    # df.to_csv(csv_file_path, index=False)

    print("CSV file created successfully.")
    
    print("A :",texts[0])
    print(f"Creating embeddings. May take some minutes...")
    vector_database.add_documents(texts)

    vector_database.persist()

if __name__ == "__main__":
    create_vector_database()
