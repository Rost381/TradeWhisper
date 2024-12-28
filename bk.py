import os
import sys
from dotenv import load_dotenv
from tw_utils import parse_arguments

load_dotenv()
MODEL_SUFFIX = os.getenv("MODEL_SUFFIX", None)

if  MODEL_SUFFIX == '' or MODEL_SUFFIX == "CMD": 
    args = parse_arguments()
    MODEL_SUFFIX = args.suffix
    if MODEL_SUFFIX == '' :
        print("Need model suffix. Exit..")
        sys.exit(0) 

for i in range(3):
    print(f"{MODEL_SUFFIX} - {i}")