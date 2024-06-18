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
from mdc import MDC
from math import ceil
from openai import OpenAI
import logging
from typing import Any
import argparse
from openai import OpenAI
from datasets import Dataset
from transformers import AutoTokenizer
import random
from math import ceil


warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "dburl")

source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')

chunk_size = 1024
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

def process_documents(ignored_files: List[str] = []) -> List[str]:
    """
    Load documents and split in chunks
    """
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)

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

def get_args() -> argparse.Namespace:
    """
    Parses and returns the arguments specified by the user's command
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--datapath", type=str, default="", help="The path at which the document is located")
    parser.add_argument("--output", type=str, default="./", help="The path at which to save the dataset")
    parser.add_argument("--distractors", type=int, default=3, help="The number of distractor documents to include per data point / triplet")
    parser.add_argument("--p", type=float, default=1.0, help="The percentage that the oracle document is included in the context")
    parser.add_argument("--questions", type=int, default=5, help="The number of data points / triplets to generate per chunk")
    parser.add_argument("--chunk_size", type=int, default=512, help="The size of each chunk in number of tokens")
    parser.add_argument("--doctype", type=str, default="pdf", help="The type of the document, must be one of the accepted doctypes", choices=["pdf", "txt", "json", "api"])
    parser.add_argument("--openai_key", type=str, default=None, help="Your OpenAI key used to make queries to gemma or llama2")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-ada-002", help="The embedding model to use to encode documents chunks (text-embedding-ada-002, ...)")
    parser.add_argument("--completion-model", type=str, default="llama2", help="The model to use to generate questions and answers (gemma, llama2, ...)")

    args = parser.parse_args()
    return args

def strip_str(s: str) -> str:
    """
    Helper function for helping format strings returned by llama2.
    """
    l, r = 0, len(s)-1
    beg_found = False
    for i in range(len(s)):
        if s[i].isalpha():
            if not beg_found:
                l = i
                beg_found = True
            else:
                r = i 
    r += 2
    return s[l:min(r, len(s))]

def encode_question_gen(question: str, chunk: Any) -> list[str]:
    """
    Encode multiple prompt instructions into a single string for the general case (`pdf`, `json`, or `txt`).
    """
    
    prompts = []
        
    prompt = """
        Question: {question}\nContext: {context}\n
        Answer this question using the information given in the context above. Here is things to pay attention to: 
        - First provide step-by-step reasoning on how to answer the question. 
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
        - End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct.
    """.format(question=question, context=str(chunk))
    prompts.append({"role": "system", "content": "You are a helpful question answerer who can provide an answer given a question and relevant context."})
    prompts.append({"role": "user", "content": prompt})
    return prompts

def generate_label(client: OpenAI, question: str, context: Any, model: str = None) -> str | None:
    """
    Generates the label / answer to `question` using `context` and llama2.
    """
    question = encode_question_gen(question, context)
    response = client.chat.completions.create(
        model='llama2',
        messages=question,
        n=1,
        temperature=0
    )
    response = response.choices[0].message.content
    return response

def generate_instructions_gen(client: OpenAI, chunk: Any, x: int = 5, model: str = None) -> list[str]:
    """
    Generates `x` questions / use cases for `chunk`. Used when the input document is of general types 
    `pdf`, `json`, or `txt`.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), generate %s example questions a user could ask and would be answered using information from the chunk. For example, if the given context was a Wikipedia paragraph about the United States, an example question could be 'How many states are in the United States?'" % (x)},
            {"role": "system", "content": "The questions should be able to be answered in a few words or less. Include only the questions in your response."},
            {"role": "user", "content": str(chunk)}
        ]
    )
    print("generate_instructions_gen generate_instructions_gen", response)
    queries = response.choices[0].message.content.split('\n')
    queries = [strip_str(q) for q in queries]
    queries = [q for q in queries if any(c.isalpha() for c in q)]

    return queries 

def add_chunk_to_dataset(
    client: OpenAI,
    chunks: list[str], 
    chunk: str, 
    x: int = 5, 
    num_distract: int = 3, 
    p: float = 0.8,
    model: str = None
) -> None:
    """
    Given a chunk, create {Q, A, D} triplets and add them to the dataset.
    """
    global ds
    i = chunks.index(chunk)
    print("Begin generate_instructions_gen", chunk)
    qs = generate_instructions_gen(client, chunk, x, model)
    print("End generate_instructions_gen")
    for q in qs:
        datapt = {
            "id": None,
            "type": None,
            "question": None,
            "context": None,
            "oracle_context": None,
            "cot_answer": None
        }

        datapt["id"] = f"seed_task_{0 if not ds else ds.num_rows}"
        datapt["type"] = "general"
        datapt["question"] = q

        # add num_distract distractor docs
        docs = [chunk]
        indices = list(range(0, len(chunks)))
        indices.remove(i)
        for j in random.sample(indices, num_distract):
            docs.append(chunks[j])
        # decides whether to add oracle document
        oracle = random.uniform(0, 1) < p
        if not oracle:
            docs[0] = chunks[random.sample(indices, 1)[0]]
        random.shuffle(docs)

        d = {
            "title": [],
            "sentences": []
        }

        d["title"].append(["placeholder_title"]*(num_distract+1))
        d["sentences"].append(docs)
        datapt["context"] = d
        datapt["oracle_context"] = chunk

        # add answer to q
        datapt["cot_answer"] = generate_label(client, q, chunk, model=model) 

        # construct model instruction 
        context = ""
        for doc in docs:
            context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
        context += q
        datapt["instruction"] = context

        # add to dataset
        if not ds:
            # init ds
            datapt["id"] = [datapt["id"]]
            datapt["type"] = [datapt["type"]]
            datapt["question"] = [datapt["question"]]
            datapt["context"] = [datapt["context"]]
            datapt["oracle_context"] = [datapt["oracle_context"]]
            datapt["cot_answer"] = [datapt["cot_answer"]]
            datapt["instruction"] = [datapt["instruction"]]
            ds = Dataset.from_dict(datapt)
        else:
            ds = ds.add_item(datapt)

def raft_dataset_documents(documents):
    # client = build_openai_client(
    #     api_key=OPENAPI_API_KEY,
    # )
    print("Begin raft_dataset_documents")
    global ds
    ds = None
    args = get_args()
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',
    )
    CHUNK_SIZE = args.chunk_size
    NUM_DISTRACT_DOCS = args.distractors
    
    chunks = [chunk.page_content for chunk in documents]

    num_chunks = len(chunks)
    print("Begin chunks", chunks[0])
    for i, chunk in enumerate(chunks):
        perc = ceil(i / num_chunks * 100)
        with MDC(progress=f"{perc}%"):
            add_chunk_to_dataset(client, chunks, chunk, args.questions, NUM_DISTRACT_DOCS, model='llama2')
    
    # Save as .arrow format
    ds.save_to_disk(args.output)
    
    # Save as .jsonl format
    ds.to_json(args.output + ".jsonl")

def load_texts():
    with open('stored_texts.pkl', 'rb') as file:
        texts = pickle.load(file)
    return texts

# Create vector database
def create_vector_database():
    args = get_args()
    
    embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
    vector_database = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )

    collection = vector_database.get()
    texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
    
    with open('stored_texts.pkl', 'wb') as file:
        pickle.dump(texts, file)
    
    raft_dataset_documents(texts)
    print("Json file created successfully.")

    print(f"Creating embeddings. May take some minutes...")
    vector_database.add_documents(texts)

    vector_database.persist()

def main():
    global ds
    create_vector_database()
    

if __name__ == "__main__":
    with MDC(progress="0%"):
        main()
