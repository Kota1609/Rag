# import required dependencies
# https://docs.chainlit.io/integrations/langchain
import os
# from langchain import hub
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import gc 
from langchain_nomic import NomicEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
# DB_DIR: str = os.path.join(ABS_PATH, "zephyr-dburl")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "dburl-old")

# Set up RetrievelQA model
# rag_prompt_mistral = hub.pull("rlm/rag-prompt-mistral")

template = '''
<s> [INST] Provider complete answer. You are an assistant for question-answering tasks. USE ONLY THE PROVIDED CONTEXT to answer the question. DO NOT USE ANY OTHER knowledge or information.  [/INST] </s>

[INST]
Context: {context}

Question: {question}

Answer: [/INST]
'''
rag_prompt_mistral = PromptTemplate(
    template=template, 
    input_variables=[
        'context', 
        'question',
    ]
)

def load_model():
    llm = Ollama(
        model="llama3:latest",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm

def retrieval_qa_chain(llm, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": rag_prompt_mistral},
        return_source_documents=True
    )
    return qa_chain

def qa_bot():
    llm = load_model()
    # DB_PATH = DB_DIR
    # embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    #embeddings = NomicEmbeddings(model="nomic-embed-text-v1")
    embeddings = HuggingFaceEmbeddings(model_name="Salesforce/SFR-Embedding-Mistral")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    qa = retrieval_qa_chain(llm, vectorstore)
    gc.collect()
    return qa

@cl.on_chat_start ## event handler - triggers when chat session starts
async def start():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    chain = qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Documents using Ollama (gemma model) and LangChain."
    )
    await welcome_message.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    cb.answer_reached = True
    print(f"chain: {chain}, callbacks: {[cb]}")
    res = await chain.acall(message.content, callbacks=[cb])

    answer = res["result"]
    #answer = answer.replace(".", ".\n")
    source_documents = res["source_documents"]
    # print("Source documents length : ", len(source_documents))

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
