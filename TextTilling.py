from typing import Sequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import nltk
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import gaussian_filter1d


def load_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def normalize_text(text: str) -> str:
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_sentences(text: str) -> list[str]:
    return nltk.sent_tokenize(text, language='russian')


def embed_sentences(sentences: list[str], model_name='cointegrated/rubert-tiny2') -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(sentences)


def compute_smoothed_similarities(embeddings: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    similarities = [
        cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        for i in range(len(embeddings) - 1)
    ]
    return gaussian_filter1d(similarities, sigma=sigma)


def find_valleys(smoothed: np.ndarray) -> list[int]:
    valleys = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
            valleys.append(i)
    return valleys


def split_into_chunks(sentences: list[str], valleys: list[int]) -> list[str]:
    chunks = []
    start = 0
    for v in valleys:
        chunks.append(' '.join(sentences[start:v+1]))
        start = v + 1
    chunks.append(' '.join(sentences[start:]))
    return chunks


def save_chunks(chunks: list[str], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            chunk_sentences = nltk.sent_tokenize(chunk, language="russian")
            f.write(f"\n--- Чанк {i + 1} | {len(chunk_sentences)} предложений ---\n")
            for j, sent in enumerate(chunk_sentences, 1):
                f.write(f"[{j}] {sent}\n")


def save_plot(smoothed: Sequence[float], valleys: list[int], output_path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed, label='Сглаженное сходство')
    plt.scatter(valleys, [smoothed[i] for i in valleys], color='red', label='Долины (границы)')
    plt.title("Text Tiling. Тематические долины")
    plt.xlabel("Переход между предложениями")
    plt.ylabel("Косинусное сходство")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    nltk.download('punkt')

    raw_text = load_text("./text-example.txt")
    text = normalize_text(raw_text)
    sentences = tokenize_sentences(text)
    embeddings = embed_sentences(sentences)
    smoothed = compute_smoothed_similarities(embeddings)
    valleys = find_valleys(smoothed)
    chunks = split_into_chunks(sentences, valleys)

    save_chunks(chunks, "extracted_scenes.txt")
    save_plot(smoothed, valleys, "scene_boundaries.png")


if __name__ == "__main__":
    main()
