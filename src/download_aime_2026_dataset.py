from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset

ds = load_dataset("MathArena/aime_2026")