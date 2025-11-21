import sys
import dspy
import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_LLM_ENDPOINT = os.getenv("DATABRICKS_LLM_ENDPOINT")
DATABRICKS_TOKEN=os.getenv("DATABRICKS_TOKEN")


class DspyHelper():
    
    @classmethod
    def init(cls):
        lm = dspy.LM(
            f"databricks/{DATABRICKS_LLM_ENDPOINT}",         
            api_key = DATABRICKS_TOKEN,
            api_base = f"{DATABRICKS_HOST}/serving-endpoints"
        )
        dspy.configure(lm=lm)
        logger.info(f"Dspy initialized with Databricks LLM endpoint: {DATABRICKS_LLM_ENDPOINT}")
        
    
    @classmethod
    def init_kimi(cls):
        import os
        lm = dspy.LM(
            model="openai/moonshot-v1-8k",
        # kimi = dspy.OpenAI(
            api_key=os.getenv("MOONSHOT_API_KEY"),
        #     model="kimi-k2-turbo-preview",   # k2 模型
            api_base="https://api.moonshot.cn/v1"
        )
        dspy.configure(lm=lm)
        logger.info(f"Dspy initialized with Kimi LLM endpoint")