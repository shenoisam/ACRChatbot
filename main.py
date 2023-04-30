import openai
import os
from dotenv import load_dotenv
#load_dotenv()
from LangChainModel import Model
from AlpacaModel import AlpacaModel
from flask import Flask, render_template, request

# Based off of: https://medium.com/@kumaramanjha2901/building-a-chatbot-in-python-using-chatterbot-and-deploying-it-on-web-7a66871e1d9b

# Set the maximum number of tokens to generate in the response
max_tokens = 1024


app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return runModel(userText)

def runModel(query):
    #AlpacaModel().generateIndex()

    AlpacaModel().run(query)


if __name__ == "__main__":
    runModel("A 56-year-old woman presents with a history of large palpable left breast mass.")