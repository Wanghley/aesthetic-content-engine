import pandas as pd
from litellm import completion
import os
from dotenv import load_dotenv

load_dotenv()

class ContentStrategist:
    def __init__(self, model_alias="gpt-4o"):
        self.model = model_alias
        
    def _get_api_params(self):
        """Resolves custom endpoints based on the model provider."""
        params = {"model": self.model}
        
        # If using a LiteLLM proxy (custom host)
        if self.model.startswith("openai") and os.getenv("LITELLM_PROXY_URL"):
            params["api_base"] = os.getenv("LITELLM_PROXY_URL")
            params["api_key"] = os.getenv("LITELLM_API_KEY")
            
        # If using Ollama on a custom host/port
        elif self.model.startswith("ollama") and os.getenv("OLLAMA_BASE_URL"):
            params["api_base"] = os.getenv("OLLAMA_BASE_URL")
            
        return params

    def generate_viral_plan(self, manifest_path):
        df = pd.read_pickle(manifest_path)
        
        # Summarize top content for the AI context
        # Filter for top 20% aesthetic photos to keep context window small/focused
        top_slice = df[df['aesthetic_score'] > df['aesthetic_score'].quantile(0.8)]
        
        prompt = f"""
        Act as a Creative Director. 
        Context: User is a Duke Engineer moving to a Rice PhD. 
        Projects: Aura (Edge AI), Arlo (Personal AI).
        Data: {len(df)} total images, {len(top_slice)} high-aesthetic candidates.
        
        Task: Create 3 Instagram "Hook" ideas and carousel structures. 
        Use a 'Dev-Changelog' aesthetic for one of them.
        """

        # LiteLLM takes the dynamic params dictionary
        api_params = self._get_api_params()
        response = completion(
            **api_params,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content