import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent  # React agent in 0.3+
import os 
from dotenv import load_dotenv

# -------------------------
# Load env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# -------------------------
# Tool setup
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

tools = [search, arxiv, wiki]

# -------------------------
st.title("LangChain - chat with search")
st.sidebar.header("Settings")

api_key_input = st.sidebar.text_input("Enter your Groq API Key", type="password")
if api_key_input:
    api_key = api_key_input

# -------------------------
# Session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant","content":"Hi, I am a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg['content'])

# -------------------------
# User input
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state["messages"].append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", streaming=True)

    # Create agent
    search_agent = create_agent(model=llm, tools=tools)

    # Run agent with .run() instead of invoke()
    assistant_reply = search_agent.run(prompt)

    # Save & display assistant reply
    st.session_state["messages"].append({"role":"assistant","content":assistant_reply})
    st.chat_message("assistant").write(assistant_reply)



#code wont work as a lot of the features used here have been depreceated
