from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    top_p=0.1,
    max_tokens=1700,
    api_key=os.getenv("OPENAI_API_KEY")
)

