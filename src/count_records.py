from dbfread import DBF
import pandas as pd

for name in ["exped", "members", "peaks"]:
    t = DBF(f"data/raw/HIMDATA/{name}.DBF", load=False)
    print(name, t.header.numrecords, len(t.fields))




# 2. Не найняті учасники (до жодних інших фільтрів)
t = DBF("data/raw/HIMDATA/members.DBF", load=False, char_decode_errors="replace")
members = pd.DataFrame(iter(t), columns=["MYEAR", "HIRED"])

members["MYEAR"] = pd.to_numeric(members["MYEAR"], errors="coerce")  
mask = members["MYEAR"].between(1990, 2025) & (members["HIRED"] == False)

print("MYEAR не число:", members["MYEAR"].isna().sum()) 
print("HIRED порожній:", members["HIRED"].isna().sum())   
print("не найнятих 1990–2025:", mask.sum())               