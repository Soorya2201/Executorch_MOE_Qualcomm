"""Build a --calib_samples JSON for ExecuTorch's Qualcomm LLM export without lm_eval.

Mixes wikitext-2 validation passages (from the local HF datasets cache) with a spread of
instruction / code / math / reasoning conversations, in the HuggingFace chat format that
examples/qualcomm/oss_scripts/llama/README.md documents for --calib_samples.
"""
import argparse
import json

from datasets import load_dataset

INSTRUCT = [
    ("What is the capital of France, and why is it historically important?",
     "The capital of France is Paris. It has been the political and cultural centre of the country for centuries, home to the monarchy, the French Revolution, and today the national government."),
    ("Write a Python function that returns the n-th Fibonacci number.",
     "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nIt runs in O(n) time and O(1) memory."),
    ("Solve for x: 3x + 7 = 22.",
     "Subtract 7 from both sides: 3x = 15. Divide by 3: x = 5."),
    ("Explain what a neural processing unit (NPU) is in two sentences.",
     "An NPU is a processor specialised for the matrix and vector math used by neural networks. It runs AI models faster and with less power than a general-purpose CPU."),
    ("Summarise the plot of Romeo and Juliet in three sentences.",
     "Two young lovers from feuding families in Verona marry in secret. A chain of misunderstandings and a fatal duel leads Romeo to believe Juliet is dead. Both die by suicide, and their deaths finally reconcile their families."),
    ("Translate 'Good morning, how are you?' into Spanish and German.",
     "Spanish: Buenos días, ¿cómo estás? German: Guten Morgen, wie geht es dir?"),
    ("Give me three tips for writing clear technical documentation.",
     "1. Start with what the reader is trying to do. 2. Show a complete, runnable example early. 3. Keep one idea per paragraph and define terms on first use."),
    ("If a train travels 180 km in 2.5 hours, what is its average speed?",
     "Average speed = distance / time = 180 km / 2.5 h = 72 km/h."),
    ("What is the difference between a list and a tuple in Python?",
     "Lists are mutable, so you can change, add or remove elements; tuples are immutable once created. Tuples are hashable when their contents are, so they can be dictionary keys."),
    ("Write a short haiku about autumn rain.",
     "Cold rain on red leaves\nthe gutter hums a low song\nsummer washes out"),
    ("Is 221 a prime number? Show your reasoning.",
     "No. 221 = 13 x 17, so it has divisors other than 1 and itself."),
    ("Explain recursion to a ten-year-old.",
     "Recursion is when a task is solved by doing a smaller version of the same task, like opening a set of nesting dolls until you reach the smallest one, then putting them back together."),
    ("Write a SQL query that returns the ten highest-paid employees.",
     "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 10;"),
    ("What causes the seasons on Earth?",
     "Earth's axis is tilted about 23.5 degrees. As Earth orbits the Sun, each hemisphere is tilted toward the Sun for part of the year, receiving more direct sunlight and longer days, which makes summer."),
    ("Compare TCP and UDP briefly.",
     "TCP is connection-oriented and guarantees ordered, reliable delivery with retransmission. UDP is connectionless and faster but offers no delivery or ordering guarantees, which suits streaming and games."),
    ("Draft a polite email asking to reschedule a meeting to Thursday.",
     "Hi Alex,\n\nWould it be possible to move our meeting to Thursday at the same time? Something came up on my side. Thank you for understanding.\n\nBest regards,\nSam"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--num_wiki", type=int, default=16)
parser.add_argument("--num_instruct", type=int, default=len(INSTRUCT))
parser.add_argument("--max_chars", type=int, default=3000, help="truncate wikitext passages")
args = parser.parse_args()

wiki = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="validation")
samples = []
for row in wiki:
    text = row["page"].strip()
    if len(text) < 1500:
        continue
    samples.append({"messages": [
        {"role": "user", "content": "Continue this encyclopedia article."},
        {"role": "assistant", "content": text[:args.max_chars]},
    ]})
    if len(samples) >= args.num_wiki:
        break

for user, assistant in INSTRUCT[:args.num_instruct]:
    samples.append({"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]})

with open(args.out, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=1)
print(f"wrote {len(samples)} samples ({args.num_wiki} wikitext + {min(args.num_instruct, len(INSTRUCT))} instruct) to {args.out}")
