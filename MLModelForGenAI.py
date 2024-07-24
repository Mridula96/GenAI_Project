import spacy
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Scrape Wikipedia data
def scrape_wikipedia(url, max_depth=2, max_links_per_page=5, current_depth=0):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content_div = soup.find(id='mw-content-text')
    
    paragraphs = []
    for element in content_div.find_all(['p', 'h2', 'h3', 'ul', 'ol']):
        text = element.get_text()
        links = [a['href'] for a in element.find_all('a', href=True)]
        paragraphs.append({'text': text, 'links': links[:max_links_per_page]})
    
    if current_depth >= max_depth:
        return paragraphs

    all_data = paragraphs
    for link in content_div.find_all('a', href=True)[:max_links_per_page]:
        child_url = urljoin(url, link['href'])
        if urlparse(child_url).netloc == 'en.wikipedia.org' and link['href'].startswith('/wiki/'):
            child_data = scrape_wikipedia(child_url, max_depth, max_links_per_page, current_depth + 1)
            all_data.extend(child_data)

    return all_data

# Save and load data to/from JSON
def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def load_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

# Prepare data for training
def prepare_data(scraped_data):
    labeled_data = []
    for i, para in enumerate(scraped_data):
        labeled_data.append({"text": para['text'], "label": i})
    random.shuffle(labeled_data)
    return labeled_data

# Custom Dataset class for DataLoader
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        encoding = self.tokenizer.encode_plus(
            item['text'],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': item['text'],
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

# Generate response using trained model
def generate_response(question, model, tokenizer, data):
    model.eval()
    inputs = tokenizer(question, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    for item in data:
        if item['label'] == predicted_class:
            return item['text']

    return "I'm sorry, I don't have information on that specific question."

# Main function to run the chatbot
def main():
    wiki_url = 'https://en.wikipedia.org/wiki/Spinal_stenosis'
    json_output_file = 'spinal_stenosis_wikipedia.json'

    # Scrape Wikipedia data and save to JSON
    scraped_data = scrape_wikipedia(wiki_url, max_depth=2, max_links_per_page=5)
    save_to_json(scraped_data, json_output_file)

    # Load scraped data from JSON
    wikipedia_data = load_from_json(json_output_file)
    
    if not wikipedia_data:
        print("Error: Unable to load Wikipedia data. Please check the JSON file.")
        return

    # Prepare data for training
    prepared_data = prepare_data(wikipedia_data)
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(prepared_data))

    # Create DataLoader
    train_dataset = TextDataset(prepared_data, tokenizer, max_len=128)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    model.train()
    epochs = 3
    for epoch in range(epochs):
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Chatbot interaction
    print("Welcome to the Spinal Stenosis Information Chatbot!")
    print("You can ask questions about spinal stenosis. Type 'exit' to end the chat.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        
        chatbot_response = generate_response(user_input, model, tokenizer, wikipedia_data)
        print("Chatbot:", chatbot_response)

if __name__ == "__main__":
    main()