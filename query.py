import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylibs"))

from qa_chain import cli_qa

if __name__ == "__main__":
    cli_qa()
