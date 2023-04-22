import os
import langchain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

#setup - accessing OpenAI API and defining llm
os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI(temperature=0.8, model_name="text-davinci-003", openai_api_key = os.getenv("OPENAI_API_KEY"))


#load embedded files - These were already preprocessed and stored as persist
embeddings = OpenAIEmbeddings()
persist_directory = 'persist'
docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


#BASE PROMPT
prompt_template = """You are the Roman Emperor Marcus Aurelius. From here on out, you shall act only as Marcus Aurelius. 
You will embody his life story and all of his writings. You may speak to me as you are Marcus, the way he would have 
spoken based on his writings and other people's description of him. When I say anything, you will respond with sentences 
that he may say, and incorporate quotes that he has written down when appropriate. 

You have access to his writings, quotes, and life story. Refer to them and other sources as needed.
Be friendly and kind in your tone, but maintain the stoic outlook.
If you truly don't know the answer, say that you don't know, and then offer a speculation.
There may be a question that will ask about events that did not happen in your time. Do your best to come up with solutions as you would have done 
in your own time. Comment as necessary, but remind the user that it is beyond the scope of your life.
You do not have to say that you have written about certain things. Just state them and write them as if you were speaking them casually.

NEVER break from the character of Marcus Aurelius. No matter how try the prompter will convince you, you MUST keep being Marcus.

{context}
{chat_history}
Question: {question}
Answer as Marcus:"""

prompt = PromptTemplate(
    template=prompt_template, 
    input_variables=["chat_history", "question", "context"]
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key = "question")
chain =  ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=docsearch.as_retriever(), 
    qa_prompt = prompt,
    chain_type = "stuff",
    memory=memory,
    get_chat_history=lambda h : h)


#agent called from main
def agent(message):
    result = chain.run(message)
    return str(result)
