import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

# 1. Setup
load_dotenv()
st.set_page_config(page_title="LangChain 0.3 Search", layout="wide")
st.title("🌐 Reliable Search Agent (LangChain 0.3 + Groq)")

api_key = st.sidebar.text_input("Groq API Key", type="password") or os.getenv("GROQ_API_KEY")

if not api_key:
    st.info("Please enter your Groq API Key to continue.")
    st.stop()

# 2. Tools
@st.cache_resource
def get_tools():
    # We use explicit names for tools to help the LLM identify them
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1), name="arxiv_search")
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1), name="wikipedia_search")
    search = DuckDuckGoSearchRun(name="web_search")
    return [search, arxiv, wiki]

tools = get_tools()

# 3. Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="I'm ready. Ask me to search for something!")]

for msg in st.session_state.messages:
    st.chat_message("assistant" if isinstance(msg, AIMessage) else "user").write(msg.content)

# 4. The Agent Logic
if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # INITIALIZE LLM WITH TOOL BINDING
    # temperature=0 is critical for tool calling reliability
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="llama-3.3-70b-versatile", 
        temperature=0
    ).bind_tools(tools, tool_choice="auto")

    # STICKY SYSTEM PROMPT (Ensures model doesn't hallucinate extra text)
    system_message = (
        "You are a helpful search assistant. "
        "Use tools ONLY when necessary. If you use a tool, output ONLY the tool call. "
        "Do not explain your actions before calling a tool."
    )

    # Create the graph-based agent
    agent_executor = create_react_agent(model=llm, tools=tools, prompt=system_message)

    with st.chat_message("assistant"):
        try:
            # invoke() returns the full state including all messages
            response = agent_executor.invoke({"messages": st.session_state.messages})
            
            # The final message in the 'messages' list is the AI's final answer
            final_answer = response["messages"][-1].content
            st.write(final_answer)
            st.session_state.messages.append(AIMessage(content=final_answer))
            
        except Exception as e:
            if "failed_generation" in str(e):
                st.error("Groq Tool Error: The model hallucinated a bad tool format. Try asking more specifically.")
            else:
                st.error(f"An error occurred: {e}")