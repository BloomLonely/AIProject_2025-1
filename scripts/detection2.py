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
import tqdm
import logging
import collections
import numpy as np
from argparse import ArgumentParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from dstc12.prompts import LABEL_CLUSTERS_PROMPT
from dstc12.utils import get_llm, DotAllRegexParser

# === Logger 설정 ===
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset_file', type=str, nargs='?', default='Dataset/AppenBanking/all.jsonl')
    parser.add_argument('preferences_file', type=str, nargs='?', default='Dataset/AppenBanking/preference_pairs.json')
    parser.add_argument('result_file', type=str, nargs='?', default='appen_banking_predicted2.jsonl') #수정할 것
    parser.add_argument('--n-clusters', type=int, default=10)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--llm-name', type=str, default="Qwen/Qwen2.5-3B")
    return parser.parse_args()


def find_second_closest_cluster(emb, centroids):
    distances = [np.linalg.norm(emb - centroid) for centroid in centroids]
    sorted_indices = np.argsort(distances)
    return sorted_indices[1]


def apply_preferences_to_clusters(utterances, utterance_embs, cluster_labels, cluster_centroids, shouldlink_pairs, cannot_link_pairs):
    logger.info("Applying preference constraints to clusters...")
    assert len(utterances) == len(cluster_labels)

    datapoint_modification_counter = collections.defaultdict(lambda: 0)
    shouldlink_applied = 0
    cannotlink_applied = 0

    utterance_cluster_mapping = collections.defaultdict(lambda: -1)
    utterance_idx_mapping = collections.defaultdict(lambda: -1)

    for idx, label in enumerate(cluster_labels):
        utterance_cluster_mapping[utterances[idx]] = label
        utterance_idx_mapping[utterances[idx]] = idx

    modified_labels = copy.deepcopy(cluster_labels)

    for a, b in shouldlink_pairs:
        if a in utterance_idx_mapping and b in utterance_idx_mapping:
            if utterance_cluster_mapping[a] != utterance_cluster_mapping[b]:
                idx_b = utterance_idx_mapping[b]
                modified_labels[idx_b] = utterance_cluster_mapping[a]
                utterance_cluster_mapping[b] = utterance_cluster_mapping[a]
                datapoint_modification_counter[idx_b] += 1
                shouldlink_applied += 1

    for a, b in cannot_link_pairs:
        if a in utterance_idx_mapping and b in utterance_idx_mapping:
            if utterance_cluster_mapping[a] == utterance_cluster_mapping[b]:
                idx_b = utterance_idx_mapping[b]
                new_cluster = find_second_closest_cluster(utterance_embs[idx_b], cluster_centroids)
                modified_labels[idx_b] = new_cluster
                utterance_cluster_mapping[b] = new_cluster
                datapoint_modification_counter[idx_b] += 1
                cannotlink_applied += 1

    total_modified = sum(datapoint_modification_counter.values())
    logger.info("Preference constraints applied.")
    logger.info(" → should_link constraints applied: %d", shouldlink_applied)
    logger.info(" → cannot_link constraints applied: %d", cannotlink_applied)
    logger.info(" → total utterances modified: %d", total_modified)
    return modified_labels




def main(utterances, linking_preferences, llm_name, n_clusters, random_state):
    embeddings = SentenceTransformer("output/mpnet-matryoshka")

    logger.info("Encoding %d utterances...", len(utterances))
    query_embeddings = embeddings.encode(utterances, convert_to_numpy=True, show_progress_bar=True)

    logger.info("Running KMeans clustering (k=%d)...", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, n_init=100, init='k-means++', random_state=random_state)
    kmeans.fit(query_embeddings)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    final_labels = apply_preferences_to_clusters(
        utterances, query_embeddings, labels, centroids,
        linking_preferences['should_link'], linking_preferences['cannot_link']
    )

    logger.info("Assigning theme labels using LLM (%s)...", llm_name)
    llm = get_llm(llm_name)
    chain = (
        LABEL_CLUSTERS_PROMPT |
        llm |
        RunnableParallel(
            theme_label=DotAllRegexParser(regex=r'<theme_label>(.*?)</theme_label>', output_keys=['theme_label']),
            theme_label_explanation=DotAllRegexParser(regex=r'<theme_label_explanation>(.*?)</theme_label_explanation>', output_keys=['theme_label_explanation'])
        )
    )

    clustered_utterances = [[] for _ in range(n_clusters)]
    for i, label in enumerate(final_labels):
        clustered_utterances[label].append(utterances[i])

    cluster_label_map = {}
    for i, cluster in tqdm.tqdm(enumerate(clustered_utterances), desc="LLM Labeling"):
        if not cluster:
            continue
        try:
            outputs_parsed = chain.invoke({'utterances': '\n'.join(cluster)})
            label = outputs_parsed['theme_label']['theme_label']
        except Exception as e:
            logger.warning(f"LLM labeling failed for cluster {i}: {e}")
            label = f"cluster_{i}"
        for utterance in cluster:
            cluster_label_map[utterance] = label

    logger.info("Labeling complete.")
    return cluster_label_map


if __name__ == '__main__':
    args = parse_args()

    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "1"

    logger.info("Reading dataset from %s", args.dataset_file)
    with open(args.dataset_file) as f:
        dataset = [json.loads(line) for line in f]

    logger.info("Extracting themed utterances...")
    themed_utterances = {
        turn['utterance']
        for dialog in dataset
        for turn in dialog['turns']
        if turn.get('theme_label') is not None
    }

    logger.info("Reading preference file from %s", args.preferences_file)
    with open(args.preferences_file) as f:
        linking_preferences = json.load(f)

    logger.info("Clustering and labeling started...")
    cluster_label_map = main(
        list(themed_utterances),
        linking_preferences,
        args.llm_name,
        args.n_clusters,
        args.random_state
    )

    logger.info("Injecting predicted labels into dataset...")
    dataset_predicted = copy.deepcopy(dataset)
    for dialog in dataset_predicted:
        for turn in dialog['turns']:
            if turn.get('theme_label') is not None:
                turn['theme_label_predicted'] = cluster_label_map.get(turn['utterance'], 'unknown')

    logger.info("Saving result to %s", args.result_file)
    with open(args.result_file, 'w') as f:
        for dialog in dataset_predicted:
            print(json.dumps(dialog), file=f)

    logger.info("✅ Finished theme detection process.")
