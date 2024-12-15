import os

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ValidationError
import json

# Set your OpenAI API key
API_KEY = os.getenv("GPT_KEY")
client = OpenAI(api_key=API_KEY)


# Pydantic model for validating responses
class GPTResponse(BaseModel):
    overall_sentiment: float   # Range [-1, 1]
    product_sentiment: float  # Range [-1, 1]
    quality_of_management: float # Range [-1, 1]
    state_of_competition: float   # Range [-1, 1]
    upcoming_events: float  # Range [-1, 1]
    semiconductor_sector: float  # Range [-1, 1]



# Function to interact with ChatGPT
def get_gpt_response(message, model="gpt-4o-mini"):
    """
    Sends a prompt to the GPT model and returns the response as a string.
    """
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": f"""
You are an assistant specialized in Intel stock analysis. Please use the same rating system for messages.
Provide JSON analyses with the following format:

    "overall_sentiment": "It tells you the overall market attitude towards Intel's prices, whether market participants anticipate an increase or decline",
    "product_sentiment": "Provides information about the status, problems and how good the products made by Intel are",
    "quality_of_management": "Informs about the status and functioning of the management board, departures, layoffs and decisions of the management board",
    "state_of_competition": "It informs about the state of competition on the market and how Intel positions itself on the market, whether it is a dominant player or whether it faces dangerous competition.",
    "upcoming_events": "Sentiment regarding future events, whether upcoming events may cause prices to rise or fall."
    "semiconductor_sector": "It determines what movements the market predicts for the semiconductor market, whether declines or increases are expected for the entire sector."
    

All values should be between -1 and 1 

Generate json for message:  {message}"""}],
            response_format=GPTResponse
        )
        return response.choices[0].message.parsed
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


# Function to generate a prompt
def generate_prompt(text):
    """
    Creates a prompt for analyzing the text without including previous responses.
    """
    return f"""
Analyze the following text about a company's stock:
{text}
Provide the analysis in this JSON format:
{{
    "company_name": "string",
    "sentiment_score": "float (range -1 to 1)",
    "relevance_score": "float (range -1 to 1)",
    "key_points": ["list of key points"],
    "comparison_summary": "string or null"
}}
    """




data = pd.read_csv("silver_df.csv")
data["day"] = pd.to_datetime(data["day"])
filter_date = pd.to_datetime('2024-06-01')
data = data[data["day"]>filter_date]
gold_layer_columns = ["day","link","overall_sentiment","product_sentiment","quality_of_management","state_of_competition","upcoming_events","semiconductor_sector"]
# ,day,url,title,content
gold_df = pd.DataFrame(columns=gold_layer_columns)  # ["day","url","title","content"]
for index, row in data.iterrows():
    text = row["content"]
    results = get_gpt_response(text)
    gold_df = pd.concat([pd.DataFrame([[row["day"], row["url"], results.overall_sentiment, results.product_sentiment,results.quality_of_management,results.state_of_competition,results.upcoming_events,results.semiconductor_sector]],
                                        columns=gold_layer_columns), gold_df], ignore_index=True)

gold_df.to_csv("gold_results.csv")
