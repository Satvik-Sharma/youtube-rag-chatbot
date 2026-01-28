from dotenv import load_dotenv
load_dotenv()
from youtube_transcript_api import YouTubeTranscriptApi
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

ytt_api = YouTubeTranscriptApi()

def getTranscripts(video_id):
    transcript = ""
    fetched_transcript = ytt_api.fetch(video_id)
    for snippet in fetched_transcript:
        transcript += snippet.text + " "
    return transcript

# Configuration
video_id = input('Enter video Id: ')
index_path = f"faiss_index_{video_id}"  # Unique per video

# Initialize embeddings (needed for both creation and loading)
embeddings = HuggingFaceEmbeddings(model='BAAI/bge-base-en-v1.5')

# Load or create vector store
if os.path.exists(index_path):
    print("Loading existing vector store...")
    vector_store = FAISS.load_local(
        index_path, 
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Creating new vector store...")
    
    # Fetch and process transcript
    transcript = getTranscripts(video_id)
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = splitter.create_documents([transcript])
    
    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save for future use
    vector_store.save_local(index_path)
    print(f"Saved vector store to {index_path}")

# Rest of your code remains the same
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

prompt_template = PromptTemplate(
    template="""Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer:""",
    input_variables=['context', 'question']
)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.2
)
model = ChatHuggingFace(llm=llm)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | model
    | StrOutputParser()
)

# Ask questions
question = input('Enter the query: ')
answer = chain.invoke(question)

print("="*80)
print("ANSWER:")
print("="*80)
print(answer)
