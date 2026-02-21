import pandas as pd
import os

def generate_html_dashboard(df, output_path="outputs/dashboard.html"):
    # Group by cluster to see the "Events" or narrative arcs
    clusters = df.groupby('cluster')
    
    html_content = f"""
    <html>
    <head>
        <title>InstaPipeline - Life Update Selector</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 40px; line-height: 1.6; }}
            h1 {{ color: #fff; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
            h2 {{ color: #007bff; margin-top: 40px; text-transform: uppercase; letter-spacing: 1px; }}
            .cluster-row {{ display: flex; overflow-x: auto; gap: 20px; padding: 20px 0; scrollbar-width: thin; }}
            .card {{ min-width: 250px; background: #181818; padding: 15px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.5); transition: transform 0.3s; }}
            .card:hover {{ transform: translateY(-5px); }}
            .img-container {{ position: relative; width: 100%; height: 200px; overflow: hidden; border-radius: 8px; margin-bottom: 10px; background: #000; }}
            img {{ width: 100%; height: 100%; object-fit: cover; cursor: pointer; }}
            .badge {{ position: absolute; top: 10px; right: 10px; background: rgba(0, 123, 255, 0.8); color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7em; }}
            .score {{ font-size: 0.9em; color: #00ff88; font-weight: bold; margin: 10px 0; }}
            .filename {{ font-size: 0.75em; color: #888; word-break: break-all; }}
            button {{ background: #007bff; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; margin-top: 10px; width: 100%; transition: background 0.2s; }}
            button:hover {{ background: #0056b3; }}
        </style>
    </head>
    <body>
        <h1>Year in Review: Content Curation</h1>
        <p>Your life indexed by AI. Select the best moments for your viral update.</p>
    """

    for cluster_id, group in clusters:
        # Sort by aesthetic score to show your best content first
        top_photos = group.sort_values('aesthetic_score', ascending=False).head(15)
        
        html_content += f"<h2>Cluster {cluster_id}: {len(group)} Memories</h2>"
        html_content += "<div class='cluster-row'>"
        
        for _, row in top_photos.iterrows():
            filename = row['filename']
            is_video = filename.lower().endswith(('.mov', '.mp4', '.avi'))
            
            # Logic: If it's a video, point to the thumbnail we generated in indexer.py
            # Otherwise, use the original data file
            display_path = f"thumbnails/{filename}.jpg" if is_video else f"../data/{filename}"
            badge_text = "VIDEO" if is_video else "IMAGE"
            
            html_content += f"""
            <div class='card'>
                <div class='img-container'>
                    <span class='badge'>{badge_text}</span>
                    <img src='{display_path}' loading="lazy" alt="{filename}">
                </div>
                <div class='score'>Score: {row['aesthetic_score']:.2f}</div>
                <div class='filename'>{filename}</div>
                <button onclick="console.log('Staged: {filename}')">Stage for Post</button>
            </div>
            """
        html_content += "</div>"

    html_content += "</body></html>"
    
    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"✅ Enhanced Dashboard generated at {output_path}")