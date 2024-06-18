import logging
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import os
import chainlit as cl
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from ingest import load_texts
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

#from constants import MODEL_ID, MODEL_BASENAME
import os
from langchain_community.llms import HuggingFacePipeline
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

#dtype = torch.bfloat16
device_type = "cuda"
persist_directory = os.environ.get("PERSIST_DIRECTORY", "dburl-old")
# embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
embeddings = HuggingFaceEmbeddings(model_name="Salesforce/SFR-Embedding-Mistral")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
  
template = """
<|start_header_id|>user<|end_header_id|>
You are an assistant for answering questions about IPM.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.
Question: {question}
Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

ALL CONVERSATION HISTORY:
{chat_history}

ANSWER:
"""


prompt = PromptTemplate(input_variables=["chat_history", "question", "context"], template=template)

condense_question_template = """
Given the conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that includes all necessary context for understanding. This rephrased question should be formulated in a way that it can be understood without needing to refer back to the conversation history, ensuring it contains all relevant details for accurate retrieval and response generation.

CONVERSATION HISTORY:
{chat_history}

FOLLOW-UP QUESTION:
{question}

REPHRASED STANDALONE QUESTION:
"""

condense_question_prompt = PromptTemplate.from_template(condense_question_template)

import transformers
import torch

def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename, resume_download=True)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            # The code supports all huggingface models that ends with GPTQ and have some variation
            # of .no-act.order or .safetensors in their HF repo.
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm

def llm_load_model():
#    model1 = AutoModelForCausalLM.from_pretrained("Kota123/ft-Gemma-7b-lora", device_map="cuda", torch_dtype=dtype)
#    model1.to("cuda")
#    model1.eval()
    llm = Ollama(
        model="llama3:latest",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    model_id = "Kota123/llama3_70b_ft"
    MODEL_ID = "Kota123/llama3_70b_ft"
    MODEL_BASENAME = None

    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

#    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
    return llm
    
@cl.on_chat_start
async def on_chat_start():
    texts = load_texts()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    bm25_retriever = BM25Retriever.from_documents(texts)
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=1)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    print("Memory :",memory)
    chain = ConversationalRetrievalChain.from_llm(
        llm_load_model(),
        chain_type="stuff",
        retriever=compression_retriever,
        # condense_question_prompt=prompt,
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory,
        return_source_documents=True,
    )
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.invoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]
    print("RES :",res)
    # print("Source chunk: ",source_documents)
    
    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"{source_doc.metadata['source']}"
            docname = source_name.replace("source_documents/", "")
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=docname)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources:\n" + "\n".join(source_names)
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
