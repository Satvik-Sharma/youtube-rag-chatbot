import streamlit as st
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

# Page configuration
st.set_page_config(
    page_title="YouTube Q&A Assistant",
    page_icon="",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF0000;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #CC0000;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Helper functions
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model='BAAI/bge-base-en-v1.5')

@st.cache_resource
def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.2
    )
    return ChatHuggingFace(llm=llm)

def get_transcripts(video_id):
    """Fetch YouTube transcript"""
    ytt_api = YouTubeTranscriptApi()
    transcript = ""
    try:
        fetched_transcript = ytt_api.get_transcript(video_id)
        for snippet in fetched_transcript:
            transcript += snippet['text'] + " "
        return transcript
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")

def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)

def load_or_create_vector_store(video_id):
    """Load existing vector store or create new one"""
    index_path = f"faiss_index_{video_id}"
    embeddings = get_embeddings()
    
    if os.path.exists(index_path):
        st.info("Loading existing vector store...")
        vector_store = FAISS.load_local(
            index_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    else:
        st.info("Creating new vector store from transcript...")
        
        with st.spinner("Fetching YouTube transcript..."):
            transcript = get_transcripts(video_id)
        
        if not transcript:
            raise Exception("No transcript found for this video")
        
        with st.spinner("Processing transcript..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            chunks = splitter.create_documents([transcript])
        
        with st.spinner("Creating embeddings..."):
            vector_store = FAISS.from_documents(chunks, embeddings)
        
        vector_store.save_local(index_path)
        st.success("Vector store created and saved")
        
        return vector_store

def create_qa_chain(vector_store):
    """Create the QA chain"""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    
    prompt_template = PromptTemplate(
        template="""Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know.

        Context: {context}

        Question: {question}

        Answer:""",
        input_variables=['context', 'question']
    )
    
    model = get_llm()
    
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | model
        | StrOutputParser()
    )
    
    return chain

# Main UI
st.title("YouTube Video Q&A Assistant")
st.markdown("Ask questions about any YouTube video transcript")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app allows you to:
    1. Enter a YouTube video ID
    2. Load the video transcript
    3. Ask questions about the content
    """)
    
    st.divider()
    
    if st.session_state.qa_history:
        st.header("Q&A History")
        for qa in reversed(st.session_state.qa_history[-5:]):
            with st.expander(f"Q: {qa['question'][:50]}..."):
                st.markdown(f"**Q:** {qa['question']}")
                st.markdown(f"**A:** {qa['answer']}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    video_id_input = st.text_input(
        "Enter YouTube Video ID",
        placeholder="e.g., dQw4w9WgXcQ"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    load_button = st.button("Load Video", use_container_width=True)

# Load video
if load_button and video_id_input:
    try:
        st.session_state.video_id = video_id_input
        st.session_state.vector_store = load_or_create_vector_store(video_id_input)
        st.session_state.chain = create_qa_chain(st.session_state.vector_store)
        st.session_state.qa_history = []
        
        st.success("Video loaded successfully")
        st.markdown(f"[Watch on YouTube](https://www.youtube.com/watch?v={video_id_input})")
        
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")

# Q&A Section
if st.session_state.chain:
    st.divider()
    st.subheader("Ask a Question")
    
    question = st.text_area(
        "Enter your question:",
        height=100
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        ask_button = st.button("Get Answer", use_container_width=True)
    with col2:
        clear_button = st.button("Clear History", use_container_width=True)
    
    if clear_button:
        st.session_state.qa_history = []
        st.rerun()
    
    if ask_button and question:
        with st.spinner("Thinking..."):
            answer = st.session_state.chain.invoke(question)
            st.session_state.qa_history.append({
                'question': question,
                'answer': answer
            })
            
            st.divider()
            st.markdown("### Answer")
            st.markdown(answer)
    
    if st.session_state.qa_history:
        st.divider()
        st.subheader("All Questions & Answers")
        for i, qa in enumerate(st.session_state.qa_history, 1):
            st.markdown(f"**Question {i}:** {qa['question']}")
            st.info(qa['answer'])
            st.markdown("---")
else:
    st.info("Enter a YouTube video ID and load the video")

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center;color:gray;'>Built with Streamlit, LangChain, and HuggingFace</div>",
    unsafe_allow_html=True
)
