from sentence_transformers import SentenceTransformer, InputExample, evaluation, losses
from sentence_transformers.losses import MatryoshkaLoss
from sentence_transformers.evaluation import LossEvaluator

from torch.utils.data import DataLoader
import torch
import json
import torch.nn as nn


class MPNET_MRL:
    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        use_mrl: bool = True,
        mrl_dim: int = 768
    ):
        self.model_name = model_path or "sentence-transformers/all-mpnet-base-v2"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.use_mrl = use_mrl
        self.mrl_dim = mrl_dim

    def encode(self, texts: list[str], batch_size: int = 32) -> list:
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        if self.use_mrl:
            embs = [emb[:self.mrl_dim] for emb in embs]
        return embs

    def fit_from_jsonl(
        self,
        jsonl_path: str,
        output_path: str = "./output/mpnet-mrl",
        batch_size: int = 16,
        epochs: int = 3,
        warmup_steps: int = 100
    ):
        train_samples = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                dialog = json.loads(line)
                for turn in dialog.get("turns", []):
                    utt = turn.get("utterance")
                    theme = turn.get("theme_label")
                    if utt and theme and "label_1" in theme:
                        train_samples.append(InputExample(texts=[utt, theme["label_1"]], label=1.0))

        print(f"✅ Loaded {len(train_samples)} training samples from {jsonl_path}")

        dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
        mnr_loss = losses.MultipleNegativesRankingLoss(self.model)
        loss = MatryoshkaLoss(
            model=self.model,
            loss=mnr_loss,
            matryoshka_dims=[768]  # 우리는 지금 768 하나만 사용
        )

        # 🧪 Evaluator 설정 (train셋 일부 사용)
        evaluator = LossEvaluator(dataloader, name="train-loss")

        self.model.fit(
            train_objectives=[(dataloader, loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True,
            evaluator=evaluator,
            evaluation_steps=100  # 배치 기준 평가 간격
        )