import spacy
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse
import random

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

def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def load_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

def generate_response(question, data):
    nlp = spacy.load('en_core_web_sm')
    keywords = [token.text.lower() for token in nlp(question) if not token.is_stop and not token.is_punct]
    best_match = None
    highest_score = 0

    for para in data:
        text = para['text'].lower()
        score = sum(keyword in text for keyword in keywords)

        if score > highest_score:
            best_match = para['text']
            highest_score = score

    return best_match if best_match else "I'm sorry, I don't have information on that specific question."

def main():
    wiki_url = 'https://en.wikipedia.org/wiki/Spinal_stenosis'
    json_output_file = 'spinal_stenosis_wikipedia.json'

    scraped_data = scrape_wikipedia(wiki_url, max_depth=2, max_links_per_page=5)
    save_to_json(scraped_data, json_output_file)

    wikipedia_data = load_from_json(json_output_file)

    if not wikipedia_data:
        print("Error: Unable to load Wikipedia data. Please check the JSON file.")
        return

    print("Welcome to the Spinal Stenosis Information Chatbot!")
    print("You can ask questions about spinal stenosis. Type 'exit' to end the chat.")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break

        chatbot_response = generate_response(user_input, wikipedia_data)
        print("Chatbot:", chatbot_response)

if __name__ == "__main__":
    main()