#!/usr/bin/env python
# coding: utf-8

# # Simple Bible Query

# ### Installing libraries

# In[ ]:


pip install PyPDF2 nltk


# In[5]:


import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
import PyPDF2


# In[ ]:


pip install torch transformers


# In[15]:


def read_pdf(file_path):
    # Open the PDF file in binary mode
    with open(file_path, 'rb') as file:
        # Create a PdfReader object from the PDF file
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Get the total number of pages in the PDF
        num_pages = len(pdf_reader.pages)
        
        # Initialise a list to store content from each page
        content = []
        
        # Iterate through each page in the PDF
        for page_num in range(num_pages):
            # Extract text content from the current page
            page = pdf_reader.pages[page_num]
            content.append(page.extract_text())
    
    # Combine the text content from all pages into a single string
    return '\n'.join(content)


# In[16]:


# Download the 'punkt' resource from NLTK (Natural Language Toolkit)
nltk.download('punkt')

# Import the sentence tokenizer from NLTK
from nltk.tokenize import sent_tokenize

def tokenize_text(text):
    # Tokenize the input text into sentences using NLTK's sentence tokenizer
    sentences = sent_tokenize(text)
    return sentences


# ### Using BERT Model

# BERT, or Bidirectional Encoder Representations from Transformers, is a powerful natural language processing model that excels in understanding contextual relationships in text by considering both left and right contexts of words, allowing it to capture intricate semantic meanings and perform various language understanding tasks with high accuracy.
# 

# In[17]:


from transformers import pipeline

# Load the question answering pipeline
qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')

def process_query_bert(text, query):
    # Use BERT for question answering
    result = qa_pipeline(context=text, question=query)
    
    # Check if the answer confidence is above a certain threshold
    if result['score'] > 0.5:
        return f"Answer: {result['answer']}"
    
    return "No relevant information found for the query."

# Usage example:
file_path = 'bible_facts2.pdf'
document_content = read_pdf(file_path)

# Prompt the user to input a question
user_query = input("Ask a question: ")

# Process the user's query using BERT and print the result
result = process_query_bert(document_content, user_query)
print(result)


# ### Adjusting the code to create a loop allowing users to keep asking questions without rerunning the entire script

# In[ ]:


from transformers import pipeline

# Load the question answering pipeline
qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')

def process_query_bert(text, query):
    """
    Process a user query using BERT-based question answering.

    Parameters:
    - text (str): The document or context for BERT to analyze.
    - query (str): The user's question to be answered.

    Returns:
    - str: The answer to the user's question or a message indicating no relevant information.
    """
    # Use BERT for question answering
    result = qa_pipeline(context=text, question=query)
    
    # Check if the answer confidence is above a certain threshold
    if result['score'] > 0.5:
        return f"Answer: {result['answer']}"
    
    return "No relevant information found for the query."

# Usage example:
file_path = 'bible_facts2.pdf'
document_content = read_pdf(file_path)

while True:
    user_query = input("Ask a question (type 'exit' to end): ")
    
    if user_query.lower() == 'exit':
        print("Exiting the question-answering loop.")
        break
    
    result = process_query_bert(document_content, user_query)
    print(result)

