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
        negsentiments.append((i+1, chapter_sentiment['neg']))
        # sentiments.append((i+2, chapter_sentiment['neg']))
    
    # Convert to DataFrame for analysis
    sentiment_df = pd.DataFrame({
        'Chapter': [x[0] for x in possentiments],
        'Positive Sentiment': [x[1] for x in possentiments],
        'Negative Sentiment': [x[1] for x in negsentiments]
    })


    # # Calculate percentages for plotting
    sentiment_df['Average'] = sentiment_df['Positive Sentiment'] - sentiment_df['Negative Sentiment']
    sentiment_df['Total Sentiment'] = sentiment_df['Positive Sentiment'] + sentiment_df['Negative Sentiment']
    # sentiment_df['Positive Percent'] = sentiment_df['Positive Sentiment'] / sentiment_df['Total']
    # sentiment_df['Negative Percent'] = sentiment_df['Negative Sentiment'] / sentiment_df['Total']

    # Plot sentiment scores
    st.subheader('Sentiment Analysis of Each Chapter')
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=sentiment_df, x='Chapter', y='Average', marker='o', label='Average Sentiment', ax=ax)
    # Plot stacked bars
    ax.bar(sentiment_df['Chapter'], sentiment_df['Positive Sentiment'], color='lightblue', label='Positive Sentiment')
    ax.bar(sentiment_df['Chapter'], sentiment_df['Negative Sentiment']*-1, color='lightcoral', label='Negative Sentiment')

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


    sentiment_df = sentiment_df.sort_values(by='Total Sentiment', ascending=False)
    # Create a Streamlit app
    st.subheader('Sentiment Analysis of Each Chapter')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot horizontal stacked bars
    bars1 = ax.barh(sentiment_df['Chapter'], sentiment_df['Positive Sentiment'], color='lightblue', label='Positive Sentiment')
    bars2 = ax.barh(sentiment_df['Chapter'], sentiment_df['Negative Sentiment'], color='lightcoral', label='Negative Sentiment', left=sentiment_df['Positive Sentiment'])

    # Set labels and title
    ax.set_title('Sentiment Analysis of Each Chapter')
    ax.set_ylabel('Chapter')
    ax.set_xlabel('Sentiment Score (%)')
    ax.axvline(x=0, color='gray', linestyle='--')  # Add a vertical line at x=0

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

    # Add percentage labels on the bars
    # for bar in bars1:
    #     width = bar.get_width()
    #     ax.annotate(f'{width:.0%}', xy=(width, bar.get_y() + bar.get_height() / 2), 
    #                 xytext=(3, 0), textcoords='offset points', ha='left', va='center')

    # for bar in bars2:
    #     width = bar.get_width()
    #     ax.annotate(f'{width:.0%}', xy=(width + sentiment_df.loc[bar.get_y(), 'Positive Sentiment'], bar.get_y() + bar.get_height() / 2), 
    #                 xytext=(-3, 0), textcoords='offset points', ha='right', va='center')

    # Show legend and grid
    ax.legend()
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)

    
    ## Grouped Bar Chart - % pos neg chapter emotions
    st.subheader('Sentiment Analysis of Each Chapter')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot grouped bar chart
    bar_width = 0.4  # Width of each bar
    chapter_indices = sentiment_df['Chapter']
    bar_positions = [ind + 1 for ind in range(len(chapter_indices))]

    # Plot bars for Positive Sentiment
    ax.bar([p - bar_width/2 for p in bar_positions], sentiment_df['Positive Sentiment'], width=bar_width, color='lightblue', label='Positive Sentiment')

    # Plot bars for Negative Sentiment
    ax.bar([p + bar_width/2 for p in bar_positions], sentiment_df['Negative Sentiment'], width=bar_width, color='lightcoral', label='Negative Sentiment')

    # Set labels and title
    ax.set_title('Sentiment Analysis of Each Chapter')
    ax.set_xlabel('Chapter')
    ax.set_ylabel('Sentiment Score (%)')
    ax.axhline(y=0, color='gray', linestyle='--')  # Add a horizontal line at y=0

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

    # Adjust x-axis ticks and labels
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(chapter_indices)

    # Show legend and grid
    ax.legend()
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)








    st.subheader('Total % Emotional Words')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=sentiment_df, x='Chapter', y='Total Sentiment', marker='o', label='Average Sentiment', ax=ax)
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