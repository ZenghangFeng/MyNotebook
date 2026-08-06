import re
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.flask import VannaFlaskApp


class ThinkOpenAI_Chat(OpenAI_Chat):
    def __init__(self, client=None, config=None):
        super().__init__(client, config)

    def submit_prompt(self, prompt, **kwargs) -> str:
        output = super().submit_prompt(prompt, **kwargs)
        print("origin output:", output)
        result = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
        print("last output:", result)
        return result


class MyVanna(ChromaDB_VectorStore, ThinkOpenAI_Chat):
    def __init__(self, client=None, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        ThinkOpenAI_Chat.__init__(self, client=client, config=config)


# 在WebUI执行程序，用户和vn进行交互 ==============================
# app = VannaFlaskApp(vn)
# app.run(port=8087)
