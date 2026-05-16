import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylibs"))

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_GLOBAL_DEVELOPMENTMODE"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

from streamlit.web import cli as stcli
from streamlit import config as st_config

st_config.set_option("global.developmentMode", False)

sys.argv = [
    "streamlit", "run",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"),
    "--global.developmentMode", "false",
]

print("=" * 50, flush=True)
print("Starting MyLibrary RAG...", flush=True)
print("=" * 50, flush=True)

stcli.main()
