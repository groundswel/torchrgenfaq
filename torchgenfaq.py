import requests
from bs4 import BeautifulSoup
import transformers
import torch

# Load a pre-trained NLG model from the transformers library
model = transformers.AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

def scrape_and_generate_qas(url, levels=5):
    # Scrape the website and extract the text content
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    
    # Use NLG model to generate questions and answers based on the text
    questions = generate_questions(text)
    answers = generate_answers(text)
    
    # Zip the questions and answers together into a list of QA pairs
    qas = list(zip(questions, answers))
    
    # If there are more levels to scrape, repeat the process for all links on the page
    if levels > 0:
        for link in soup.find_all('a'):
            href = link.get('href')
            if href.startswith('http'):
                qas += scrape_and_generate_qas(href, levels=levels-1)
    
    return qas

# Define the generate_questions and generate_answers functions
def generate_questions(text):
    questions = []
    input_dict = model.generate_questions(text)
    for question in input_dict:
        questions.append(question)
    return questions

def generate_answers(text):
    answers = []
    input_dict = model.generate_answers(text)
    for answer in input_dict:
        answers.append(answer)
    return answers

# Scrape a website and generate QA pairs
url = 'https://en.wikipedia.org/wiki/Natural_language_processing'
qas = scrape_and_generate_qas(url, levels=3)

# Convert the QA pairs to tensors and use them to fine-tune the NLG model
input_ids = []
attention_mask = []
token_type_ids = []
labels = []
for question, answer in qas:
    # Tokenize the input and label
    input_tokens = model.tokenize(question)
    label_tokens = model.tokenize(answer)
    
    # Create input and label tensors
    input_ids.append(torch.tensor([model.encode(input_tokens)]))
    attention_mask.append(torch.tensor([[1] * len(input_tokens)]))
    token_type_ids.append(torch.tensor([[0] * len(input_tokens)]))
    labels.append(torch.tensor([model.encode(label_tokens)]))

# Convert the lists to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_mask = torch.cat(attention_mask, dim=0)
token_type_ids = torch.cat(token_type_ids, dim=0)
labels = torch.cat(labels, dim=0)

# Define the loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
for epoch in range(10):
    # Forward pass
    logits = model(input_ids, attention_mask, token_type_ids)
    loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f'Loss at epoch {epoch}: {loss.item()}')

# Save the fine-tuned model
model.save_pretrained('./nlg_model')

# Define a function to ask questions and get answers from the model
def ask_question(question):
    # Tokenize the input
    input_tokens = model.tokenize(question)
    
    # Create an input tensor
    input_ids = torch.tensor([model.encode(input_tokens)])
    attention_mask = torch.tensor([[1] * len(input_tokens)])
    token_type_ids = torch.tensor([[0] * len(input_tokens)])
    
    # Get the logits for the input
    logits = model(input_ids, attention_mask, token_type_ids)
    
    # Get the index of the most likely answer
    answer_index = logits.argmax().item()
    
  # Decode the answer from the model's vocabulary
answer = model.decode([answer_index])

# Ask the user for feedback on the answer
feedback = input(f'Did you find this answer satisfactory? (y/n) {answer}')

if feedback == 'n':
    # Get the correct answer from the user
    correct_answer = input('What is the correct answer?')
    
    # Tokenize the correct answer
    correct_answer_tokens = model.tokenize(correct_answer)
    
    # Create a tensor for the correct answer
    correct_answer_ids = torch.tensor([model.encode(correct_answer_tokens)])
    correct_attention_mask = torch.tensor([[1] * len(correct_answer_tokens)])
    correct_token_type_ids = torch.tensor([[0] * len(correct_answer_tokens)])
    
    # Use the correct answer to update the model's logits
    logits[:, answer_index] = model(correct_answer_ids, correct_attention_mask, correct_token_type_ids)[:, 0]
    
    # Calculate the new loss with the updated logits
    new_loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    
    # Backward pass
    new_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    


    print(f'Loss after feedback: {new_loss.item()}')

    # Save the fine-tuned model
    model.save_pretrained('./nlg_model')

    # Define a function to ask questions and get answers from the model
    def ask_question(question):
        # Tokenize the input
        input_tokens = model.tokenize(question)

        # Create an input tensor
        input_ids = torch.tensor([model.encode(input_tokens)])
        attention_mask = torch.tensor([[1] * len(input_tokens)])
        token_type_ids = torch.tensor([[0] * len(input_tokens)])

        # Get the logits for the input
        logits = model(input_ids, attention_mask, token_type_ids)

        # Get the index of the most likely answer
        answer_index = logits.argmax().item()

        # Decode the answer from the model's vocabulary
        answer = model.decode([answer_index])

        return answer



