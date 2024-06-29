import re


def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def split_into_chapters(text):
    split_text = re.split(r'(Chapter \d+)', text)
    chapter_texts = [split_text[i] + split_text[i + 1] for i in range(1, len(split_text) - 1, 2)]
    # Skip the first 15 chapters which are part of the index
    chapters = chapter_texts[15:]
    return chapters

def preprocess_text(text):
    # Remove extra spaces and newlines
    text = text.replace('\n', ' ').replace('  ', ' ').strip()

    return text


def split_into_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def extract_dialogue_and_nondialogue(sentences):
    dialogue = []
    non_dialogue = []
    for sentence in sentences:
        if '"' in sentence:
            parts = re.split(r'("[^"]*")', sentence)
            for part in parts:
                if '"' in part:
                    dialogue.append(part.strip())
                else:
                    non_dialogue.append(part.strip())
        else:
            non_dialogue.append(sentence.strip())
    return dialogue, non_dialogue