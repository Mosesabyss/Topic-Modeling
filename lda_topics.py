import pandas as pd
import re
from typing import List, Tuple
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer


def preprocess_text(texts: pd.Series) -> List[List[str]]:
    """Preprocess a series of texts into token lists."""
    lemmatizer = WordNetLemmatizer()
    processed = []
    for doc in texts.dropna():
        doc = doc.lower()
        doc = re.sub(r"[^a-zA-Z\s]", "", doc)
        tokens = [lemmatizer.lemmatize(token) for token in doc.split() if token not in STOPWORDS]
        processed.append(tokens)
    return processed


def train_lda(docs: List[List[str]], num_topics: int = 10) -> Tuple[LdaModel, corpora.Dictionary, List[List[tuple]]]:
    """Train an LDA model and return the model, dictionary and corpus."""
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(text) for text in docs]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
    return lda, dictionary, corpus


def top_keywords_per_topic(model: LdaModel, num_words: int = 10) -> List[str]:
    """Return top keywords for each topic."""
    topics = []
    for t in range(model.num_topics):
        words = model.show_topic(t, topn=num_words)
        topic_words = ", ".join([w for w, _ in words])
        topics.append(topic_words)
    return topics


def sample_comments(df: pd.DataFrame, corpus: List[List[tuple]], model: LdaModel, dictionary: corpora.Dictionary) -> List[List[str]]:
    """Return two sample comments for each topic."""
    topic_comments = [[] for _ in range(model.num_topics)]
    for i, bow in enumerate(corpus):
        topic_prob = model.get_document_topics(bow)
        if not topic_prob:
            continue
        top_topic = max(topic_prob, key=lambda x: x[1])[0]
        if len(topic_comments[top_topic]) < 2:
            topic_comments[top_topic].append(df.iloc[i]["comment"])
    return topic_comments


def extract_topics(df: pd.DataFrame, num_topics: int = 10) -> pd.DataFrame:
    """Process comments and extract topics with sample comments."""
    docs = preprocess_text(df["comment"])
    lda, dictionary, corpus = train_lda(docs, num_topics)
    keywords = top_keywords_per_topic(lda)
    samples = sample_comments(df, corpus, lda, dictionary)

    data = []
    for idx in range(num_topics):
        sample = " | ".join(samples[idx]) if idx < len(samples) else ""
        data.append({
            "topic": idx,
            "keywords": keywords[idx] if idx < len(keywords) else "",
            "sample_comments": sample
        })
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage with a placeholder DataFrame
    df_simulated = pd.DataFrame({
        "comment": [
            "This movie was excellent with brilliant acting.",
            "I hated the film. It was too long and boring.",
            "The plot was thrilling and kept me on the edge of my seat.",
            "Terrible movie. Would not recommend it to anyone.",
            "Amazing cinematography and great soundtrack!",
            "The storyline lacked depth and was predictable.",
            "An absolute masterpiece of modern cinema.",
            "Waste of time. The worst movie I've seen.",
            "Outstanding performances by the entire cast.",
            "The script was poorly written and confusing."],
    })

    result_df = extract_topics(df_simulated)
    for idx, row in result_df.iterrows():
        print(f"Topic {row['topic']}: {row['keywords']}")
        for c in row['sample_comments'].split(" | "):
            print(f"  - {c}")
        print()

    print(result_df)
