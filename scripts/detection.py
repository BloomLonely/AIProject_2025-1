# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import sys

project_root = os.getcwd()
sys.path.append(os.path.join(project_root, "src"))
sys.path.append(os.path.join(project_root, "scripts"))
sys.path.append(os.path.join(project_root, "model"))

import json
import copy
import logging
import collections
import numpy as np
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from tqdm import tqdm

from mpnet import MPNET

# LLM 관련 추가
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel
from dstc12.prompts import LABEL_CLUSTERS_PROMPT
from dstc12.utils import get_llm, DotAllRegexParser

# 로그 설정
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset_file', type=str, nargs='?', default='Dataset/AppenBanking/all.jsonl')
    parser.add_argument('preferences_file', type=str, nargs='?', default='Dataset/AppenBanking/preference_pairs.json')
    parser.add_argument('result_file', type=str, nargs='?', default='appen_banking_predicted.jsonl')
    parser.add_argument('--n-clusters', type=int, default=10)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument("--llm-name", type=str, default="Qwen/Qwen2.5-3B")  # 추가
    return parser.parse_args()


def find_second_closest_cluster(emb, centroids):
    distances = [np.linalg.norm(emb - centroid) for centroid in centroids]
    return np.argsort(distances)[1]


def apply_preferences(utterances, embs, labels, centroids, should_links, cannot_links):
    logger.info("🔁 Applying preference constraints...")
    idx_map = {utt: idx for idx, utt in enumerate(utterances)}
    label_map = {utt: labels[idx_map[utt]] for utt in utterances}

    mod_labels = copy.deepcopy(labels)
    modifications = 0
    should_applied = 0
    cannot_applied = 0

    for a, b in should_links:
        if a in idx_map and b in idx_map:
            if label_map[a] != label_map[b]:
                mod_labels[idx_map[b]] = label_map[a]
                label_map[b] = label_map[a]
                should_applied += 1
                modifications += 1

    for a, b in cannot_links:
        if a in idx_map and b in idx_map:
            if label_map[a] == label_map[b]:
                new_label = find_second_closest_cluster(embs[idx_map[b]], centroids)
                mod_labels[idx_map[b]] = new_label
                label_map[b] = new_label
                cannot_applied += 1
                modifications += 1

    logger.info(f"✅ Preferences applied: should_link={should_applied}, cannot_link={cannot_applied}, total_modified={modifications}")
    return mod_labels


def generate_theme_labels(utterances, labels, n_clusters, llm_name):
    logger.info("🔎 Generating theme labels using LLM (%s)...", llm_name)
    llm = get_llm(llm_name)
    chain = (
        LABEL_CLUSTERS_PROMPT |
        llm |
        RunnableParallel(
            theme_label=DotAllRegexParser(regex=r"<theme_label>(.*?)</theme_label>", output_keys=["theme_label"]),
            theme_label_explanation=DotAllRegexParser(regex=r"<theme_label_explanation>(.*?)</theme_label_explanation>", output_keys=["theme_label_explanation"])
        )
    )

    clustered_utterances = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clustered_utterances[label].append(utterances[i])

    label_map = {}
    for i, cluster in tqdm(enumerate(clustered_utterances), desc="LLM Labeling"):
        if not cluster:
            continue
        try:
            outputs = chain.invoke({"utterances": "\n".join(cluster)})
            label = outputs["theme_label"]["theme_label"]
        except Exception as e:
            logger.warning(f"⚠️ LLM failed for cluster {i}: {e}")
            label = f"cluster_{i}"
        for utt in cluster:
            label_map[utt] = label
    return label_map


def main(utterances, preferences, n_clusters, random_state, llm_name):
    model = MPNET()
    embs = model.encode(utterances)

    logger.info("🔢 Performing KMeans clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(embs)
    labels = kmeans.labels_

    labels = apply_preferences(
        utterances, embs, labels, kmeans.cluster_centers_,
        preferences["should_link"], preferences["cannot_link"]
    )

    cluster_label_map = generate_theme_labels(utterances, labels, n_clusters, llm_name)
    logger.info("📦 Clustering and LLM labeling completed.")
    return cluster_label_map


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"📂 Loading dataset from {args.dataset_file}")
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    utterances = {
        turn["utterance"]
        for dialog in dataset
        for turn in dialog["turns"]
        if turn.get("theme_label") is not None
    }

    logger.info(f"📂 Loading preferences from {args.preferences_file}")
    with open(args.preferences_file, "r", encoding="utf-8") as f:
        preferences = json.load(f)

    cluster_label_map = main(
        list(utterances),
        preferences,
        args.n_clusters,
        args.random_state,
        args.llm_name
    )

    logger.info(f"📝 Writing predictions to {args.result_file}")
    dataset_pred = copy.deepcopy(dataset)
    for dialog in dataset_pred:
        for turn in dialog["turns"]:
            if turn.get("theme_label") is not None:
                turn["theme_label_predicted"] = cluster_label_map.get(turn["utterance"], "unknown")

    with open(args.result_file, "w", encoding="utf-8") as f:
        for dialog in dataset_pred:
            print(json.dumps(dialog, ensure_ascii=False), file=f)

    logger.info("✅ Theme detection with LLM labeling completed successfully.")
