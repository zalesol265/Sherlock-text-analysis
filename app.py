import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from preprocessing import *
from analysis import *


def main():
    st.title("Sherlock Holmes - Sentiment Analysis and Dialogue Extraction")
    
    file_path = 'sherlockholmes.txt'
    text = load_text(file_path)
    chapters = split_into_chapters(text)
    
    all_dialogue = []
    all_non_dialogue = []
    possentiments = []
    negsentiments = []
    
    for i, chapter in enumerate(chapters):
        chapter = preprocess_text(chapter)
        sentences = split_into_sentences(chapter)
        dialogue, non_dialogue = extract_dialogue_and_nondialogue(sentences)
        all_dialogue.extend(dialogue)
        all_non_dialogue.extend(non_dialogue)
        
        # Analyze sentiments
        chapter_sentiment = analyze_sentiment(chapter)
        possentiments.append((i+1, chapter_sentiment['pos']))
        negsentiments.append((i+1, chapter_sentiment['neg'] * -1))
        # sentiments.append((i+2, chapter_sentiment['neg']))
    
    # Convert to DataFrame for analysis
    sentiment_df = pd.DataFrame({
        'Chapter': [x[0] for x in possentiments],
        'Positive Sentiment': [x[1] for x in possentiments],
        'Negative Sentiment': [x[1] for x in negsentiments]
    })


    # # Calculate percentages for plotting
    sentiment_df['Average'] = sentiment_df['Positive Sentiment'] + sentiment_df['Negative Sentiment']
    sentiment_df['Total % Emotional Words'] = sentiment_df['Positive Sentiment'] - sentiment_df['Negative Sentiment']
    # sentiment_df['Positive Percent'] = sentiment_df['Positive Sentiment'] / sentiment_df['Total']
    # sentiment_df['Negative Percent'] = sentiment_df['Negative Sentiment'] / sentiment_df['Total']

    # Plot sentiment scores
    st.subheader('Sentiment Analysis of Each Chapter')
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=sentiment_df, x='Chapter', y='Average', marker='o', label='Average Sentiment', ax=ax)
    # Plot stacked bars
    ax.bar(sentiment_df['Chapter'], sentiment_df['Positive Sentiment'], color='lightblue', label='Positive Sentiment')
    ax.bar(sentiment_df['Chapter'], sentiment_df['Negative Sentiment'], color='lightcoral', label='Negative Sentiment')

    # Set labels and title
    ax.set_title('Sentiment Analysis of Each Chapter')
    ax.set_xlabel('Chapter')
    ax.set_ylabel('Sentiment Score (%)')
    ax.axhline(y=0, color='gray', linestyle='--')  # Add a horizontal line at y=0

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

    # Show legend and grid
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)



    st.subheader('Total % Emotional Words')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=sentiment_df, x='Chapter', y='Total % Emotional Words', marker='o', label='Average Sentiment', ax=ax)
    ax.set_title('Total % Emotional Words of Each Chapter')
    ax.set_xlabel('Chapter')
    ax.set_ylabel('% Emotion')
    ax.grid(True)
    ax.legend()  # Add legend to differentiate positive and negative sentiments

    st.pyplot(fig)



    # Plot sentiment scores
    st.subheader('Sentiment Analysis of Each Chapter')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot positive sentiment
    sns.lineplot(data=sentiment_df, x='Chapter', y='Positive Sentiment', marker='o', label='Positive Sentiment', ax=ax)

    # Plot negative sentiment
    sns.lineplot(data=sentiment_df, x='Chapter', y='Negative Sentiment', marker='o', label='Negative Sentiment', ax=ax)

    ax.set_title('Sentiment Analysis of Each Chapter')
    ax.set_xlabel('Chapter')
    ax.set_ylabel('Sentiment Score')
    ax.grid(True)
    ax.legend()  # Add legend to differentiate positive and negative sentiments

    st.pyplot(fig)
    
    # Distribution of sentiment scores
    st.subheader('Distribution of Sentiment Scores')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(sentiment_df['Positive Sentiment'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Sentiment Scores')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    # DataFrame for dialogue and non-dialogue
    dialogue_df = pd.DataFrame(all_dialogue, columns=['Dialogue'])
    non_dialogue_df = pd.DataFrame(all_non_dialogue, columns=['Non-Dialogue'])
    
    # Display the most positive and most negative sentences
    all_sentences = all_dialogue + all_non_dialogue
    all_sentiment = [(sentence, TextBlob(sentence).sentiment.polarity) for sentence in all_sentences]
    most_positive_sentence = max(all_sentiment, key=lambda x: x[1])
    most_negative_sentence = min(all_sentiment, key=lambda x: x[1])
    
    st.subheader('Most Positive Sentence')
    st.write(most_positive_sentence[0])
    st.write(f"Sentiment Score: {most_positive_sentence[1]}")
    
    st.subheader('Most Negative Sentence')
    st.write(most_negative_sentence[0])
    st.write(f"Sentiment Score: {most_negative_sentence[1]}")

if __name__ == "__main__":
    main()