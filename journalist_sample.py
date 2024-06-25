import streamlit as st
from textwrap import dedent
from phi.assistant import Assistant
from phi.tools.newspaper_toolkit import NewspaperToolkit
from phi.llm.openai import OpenAIChat

# Dummy credentials
USERNAME = "riskedge@123"
PASSWORD = "helloworld"

# Fixed API key (replace with your actual API key)
FIXED_API_KEY = "your_fixed_api_key_here"

# Create a simple login function
def login(username, password):
    return username == USERNAME and password == PASSWORD

def main_app(api_key):
    st.title("AI Journalist Agent 🗞️")
    st.caption("Generate high-quality articles with AI Journalist by researching, writing, and editing articles using GPT-4o.")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    writer = Assistant(
        name="Writer",
        role="Retrieves text from URLs and writes a high-quality article",
        llm=OpenAIChat(model="gpt-4o", api_key=api_key),
        description=dedent(
            """\
        You are a senior writer for the New York Times. Given a topic and a list of URLs,
        your goal is to write a high-quality NYT-worthy article on the topic.
        """
        ),
        instructions=[
            "Given a topic and a list of URLs, read each article using `get_article_text`.",
            "Then write a high-quality NYT-worthy article on the topic.",
            "Ensure the length is at least as long as a NYT cover story -- at a minimum, 15 paragraphs.",
            "Ensure you provide a nuanced and balanced opinion, quoting facts where possible.",
            "Remember: you are writing for the New York Times, so the quality of the article is important.",
            "Focus on clarity, coherence, and overall quality.",
            "Never make up facts or plagiarize. Always provide proper attribution.",
        ],
        tools=[NewspaperToolkit()],
        add_datetime_to_instructions=True,
        add_chat_history_to_prompt=True,
        num_history_messages=3,
    )

    editor = Assistant(
        name="Editor",
        llm=OpenAIChat(model="gpt-4o", api_key=api_key),
        team=[writer],
        description="You are a senior NYT editor. Given a topic, your goal is to write a NYT-worthy article.",
        instructions=[
            "Given a topic and URLs, pass the description of the topic and URLs to the writer to get a draft of the article.",
            "Edit, proofread, and refine the article to ensure it meets the high standards of the New York Times.",
            "The article should be extremely articulate and well-written.",
            "Focus on clarity, coherence, and overall quality.",
            "Ensure the article is engaging and informative.",
            "Remember: you are the final gatekeeper before the article is published.",
        ],
        add_datetime_to_instructions=True,
        markdown=True,
    )

    # Input field for the report query
    query = st.text_input("What do you want the AI journalist to write an article on?")

    if query:
        use_links = st.radio("Do you want to provide reference links?", ("Yes", "No"))

        links = []
        if use_links == "Yes":
            num_links = st.selectbox("How many reference links do you want to provide?", options=[1, 2, 3, 4, 5], index=4)
            for i in range(num_links):
                link = st.text_input(f"Enter reference link {i+1}", key=f"link_{i+1}")
                links.append(link)

        if use_links == "No" or (use_links == "Yes" and all(links)):
            with st.spinner("Processing..."):
                # Prepare the content for the writer
                links_text = "\n".join(links) if links else "No reference links provided."
                writer_instructions = f"Topic: {query}\nReference Links:\n{links_text}"

                # Get the response from the assistant
                response = editor.run(writer_instructions, stream=False)
                st.write(response)

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        main_app(FIXED_API_KEY)
    else:
        st.title("Login Page")

        # Create login form
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if login(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

if __name__ == "__main__":
    main()
