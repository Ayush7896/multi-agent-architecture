from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,    # 0.5 was too conservative for creative travel planning
    # top_p removed — don't set both temperature AND top_p together.
    # OpenAI recommends altering one, not both. top_p=0.1 was making
    # responses very repetitive and short.
    max_tokens=2000,    # bumped slightly — travel itineraries need space
    api_key=os.getenv("OPENAI_API_KEY")
)

