import pandas as pd
import os

def generate_html_dashboard(df, output_path="outputs/dashboard.html"):
    # Group by cluster to see the "Events"
    clusters = df.groupby('cluster')
    
    html_content = f"""
    <html>
    <head>
        <title>InstaPipeline - Life Update Selector</title>
        <style>
            body {{ font-family: sans-serif; background: #121212; color: white; padding: 20px; }}
            .cluster-row {{ display: flex; overflow-x: auto; gap: 15px; padding: 20px 0; border-bottom: 1px solid #333; }}
            .card {{ min-width: 200px; background: #1e1e1e; padding: 10px; border-radius: 8px; text-align: center; }}
            img {{ width: 100%; border-radius: 4px; cursor: pointer; transition: transform 0.2s; }}
            img:hover {{ transform: scale(1.05); outline: 2px solid #007bff; }}
            .score {{ font-size: 0.8em; color: #888; margin-top: 5px; }}
            h2 {{ color: #007bff; }}
        </style>
    </head>
    <body>
        <h1>Year in Review: Content Curation</h1>
        <p>Clusters identified based on visual semantics. Select photos for your carousel.</p>
    """

    for cluster_id, group in clusters:
        # Sort by aesthetic score
        top_photos = group.sort_values('aesthetic_score', ascending=False).head(10)
        
        html_content += f"<h2>Cluster {cluster_id} - Semantic Group</h2>"
        html_content += "<div class='cluster-row'>"
        
        for _, row in top_photos.iterrows():
            img_path = f"../data/{row['filename']}" # Relative to outputs folder
            html_content += f"""
            <div class='card'>
                <img src='{img_path}' onclick="alert('Added to Staging: {row['filename']}')">
                <div class='score'>Aesthetic: {row['aesthetic_score']:.2f}</div>
            </div>
            """
        html_content += "</div>"

    html_content += "</body></html>"
    
    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"✅ Dashboard generated at {output_path}")