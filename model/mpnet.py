from sentence_transformers import SentenceTransformer
import torch


class MPNET:
    def __init__(self, device: str = None):
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: list[str], batch_size: int = 32) -> list:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
