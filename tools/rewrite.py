import os

# ベースディレクトリ
base_dir = "dir"

# 削除対象の文字列
target_string = "<?xml version='1.0' encoding='utf-8'?>"

# ファイルのループ
for root, _, files in os.walk(base_dir):
    for filename in files:
        if filename.endswith(".xml"):  # XMLファイルのみを対象とする
            file_path = os.path.join(root, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            with open(file_path, 'w', encoding='utf-8') as file:
                for line in lines:
                    if target_string not in line:
                        file.write(line)

print("文字列を削除しました。")
