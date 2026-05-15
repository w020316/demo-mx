import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylibs"))
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
from streamlit.web import cli as stcli
sys.argv = ["streamlit", "run", os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")]
stcli.main()
