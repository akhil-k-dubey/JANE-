#!/usr/bin/env python3
"""Firehose Knowledge Ingestion Runner for Project Jane.

Usage:
    python3 scripts/run_knowledge_ingestion.py
    python3 scripts/run_knowledge_ingestion.py --feed "http://export.arxiv.org/rss/physics"
"""

import sys
import os
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from src.reasoning_engine.self_model import SelfModelEngine
from src.reasoning_engine.active_learner import ActiveLearner
from src.symbolic_verifier.z3_interface import Z3LogicVerifier
from src.core.knowledge_ingestor import KnowledgeIngestor


def main():
    parser = argparse.ArgumentParser(description="Firehose Knowledge Ingestor for Project Jane.")
    parser.add_argument("--feed", type=str, default=None, help="Optional arXiv RSS feed URL to parse.")
    args = parser.parse_args()

    print("=" * 65)
    print("      PROJECT JANE — FIREHOSE KNOWLEDGE INGESTION DAEMON")
    print("=" * 65)

    # Initialize components
    self_model = SelfModelEngine()
    manifold = HebbianConceptManifold(latent_dim=256)
    z3_verifier = Z3LogicVerifier()
    active_learner = ActiveLearner(manifold=manifold, z3_verifier=z3_verifier, self_model=self_model)
    ingestor = KnowledgeIngestor(active_learner=active_learner, manifold=manifold, z3_verifier=z3_verifier)

    print(f"  • Starting Concept Count: {self_model.core_concepts_count}")
    print(f"  • Starting KG Entries:   {self_model.kg_entries_count}")
    print(f"  • Symbolic Gatekeeper:   Z3 SMT Solver (Zero Pseudo-Science Policy)")
    print("-" * 65)

    # 1. Ingest Curated Scientific Corpus
    print("[Ingestion] Streaming Curated Scientific Corpus through Z3 Gate...")
    report = ingestor.ingest_curated_seed_corpus()

    print(f"  ✓ Processed: {report.total_processed} candidate concepts")
    print(f"  ✓ Accepted:  {report.accepted_count} (Formally Proven by Z3)")
    print(f"  ✗ Rejected:  {report.rejected_count} (Failed Physics Invariants)")
    print("  • Newly Wired Concepts:")
    for c in report.accepted_concepts:
        print(f"     + {c}")

    # 2. Ingest external RSS Feed if specified
    if args.feed:
        try:
            import feedparser
            print(f"\n[Ingestion] Fetching live feed from {args.feed}...")
            feed = feedparser.parse(args.feed)
            feed_entries = [{"title": e.get("title", ""), "summary": e.get("summary", "")} for e in feed.entries]
            feed_report = ingestor.ingest_raw_feed_entries(feed_entries)
            print(f"  ✓ Feed Items Accepted: {feed_report.accepted_count} / {feed_report.total_processed}")
        except ImportError:
            print("\n[Notice] 'feedparser' not installed. To stream live RSS, run: pip install feedparser")
        except Exception as e:
            print(f"\n[Notice] Could not connect to external feed: {e}")

    # Refresh self model
    self_model.load_model()
    print("-" * 65)
    print("  INGESTION SUMMARY:")
    print(f"  • Final Core Concepts:    {self_model.core_concepts_count} (+{report.accepted_count})")
    print(f"  • Final KG Entries:       {self_model.kg_entries_count}")
    print(f"  • 256D Manifold Status:   Hebbian Synapses Updated & Persisted")
    print("=" * 65)


if __name__ == "__main__":
    main()
