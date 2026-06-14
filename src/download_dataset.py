import kagglehub
from pathlib import Path

path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Downloaded to:")
print(path)

path = Path(path)

print("\nTop-level files/folders:")
for item in path.iterdir():
    print(item)