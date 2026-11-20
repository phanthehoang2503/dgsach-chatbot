import os
import subprocess
import sys


def test_real_env_wins_over_dotenv():
    env = {
        **os.environ,
        "LLM_MODEL": "from-real-env",
        "EMBEDDING_MODEL": "from-real-env-embed",
    }
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.llm, src.embeddings, os; print(os.environ['LLM_MODEL'], os.environ['EMBEDDING_MODEL'])",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "from-real-env from-real-env-embed"
