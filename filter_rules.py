import pandas as pd
import re

df = pd.read_csv("rules/process-rule-file.csv", sep="\t") # path to file with extracted rules

with open("rules/process-rule-filtered.csv", "w") as f:
    f.write("body\thead\tconfidence\trule\n")
    for index, row in df.iterrows():
        if row["confidence"] >= 0.8 and "domain" not in row["rule"] and "range" not in row["rule"]:
            # f.write(f"{row['body']}\t{row['head']}\t{row['confidence']}\t{row['rule']}\n")
            # matches = re.findall(r'(\w+)\(', row["rule"])
            matches = re.findall(r'\b\w+\s*\([^()]*\)', row["rule"])
            if "Death" in matches[0] or "Outcome" in matches[0]:
                f.write(f"{row['body']}\t{row['head']}\t{row['confidence']}\t{row['rule']}\n")

# for index, row in df.iterrows():
#     rule = row["rule"]
#     match = re.match(r'^\s*([^\s<=>]+)\s*[<=>]', rule)
#     if match:
#         result = match.group(1)
#         # if "Death" in result:
#         #     print(rule)
#         if "Outcome" in result:
#             print(rule)