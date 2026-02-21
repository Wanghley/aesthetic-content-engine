import argparse
import os
import pandas as pd
from src.indexing.indexer import LifeIndexer
from src.indexing.clusterer import cluster_life_events
from src.generation.gallery import generate_html_dashboard
from src.generation.strategist import ContentStrategist

def main():
    parser = argparse.ArgumentParser(description="InstaPipeline: Life Update Engine")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LiteLLM model alias")
    parser.add_argument("--reindex", action="store_true", help="Force re-processing of photos")
    args = parser.parse_args()

    # Paths
    data_dir = "data"
    manifest_path = "outputs/manifest.pkl"
    
    if not os.path.exists("outputs"): os.makedirs("outputs")

    # 1. Indexing (The Heavy Lifting)
    if not os.path.exists(manifest_path) or args.reindex:
        print(f"🚀 Indexing 1000+ photos in {data_dir}...")
        indexer = LifeIndexer()
        df = indexer.process_folder(data_dir)
    else:
        print(f"📂 Loading existing manifest from {manifest_path}")
        df = pd.read_pickle(manifest_path)

    # 2. Clustering (Narrative Creation)
    print("🤖 Clustering memories into 12 major life arcs...")
    df = cluster_life_events(df, n_clusters=12)
    df.to_pickle(manifest_path) # Save clusters

    # 3. Strategy (LLM Content Planning)
    print(f"✍️  Consulting the Strategist using {args.model}...")
    strategist = ContentStrategist(model_alias=args.model)
    plan = strategist.generate_viral_plan(manifest_path)
    
    with open("outputs/strategy.txt", "w") as f:
        f.write(plan)

    # 4. Gallery Generation
    generate_html_dashboard(df)
    
    print("\n" + "="*30)
    print("✅ PIPELINE COMPLETE")
    print(f"📸 Gallery:  file://{os.path.abspath('outputs/dashboard.html')}")
    print(f"📝 Strategy: outputs/strategy.txt")
    print("="*30)

if __name__ == "__main__":
    main()