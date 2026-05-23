# 生成输入demo
# 包含所有.cache->issue_sample.jsonl中记录的样本

import os
import random
import shutil

def sample_files(src_dir: str, dst_dir: str, n: int):
    # 收集所有文件路径
    all_files = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            all_files.append(os.path.join(root, f))

    sampled = random.sample(all_files, min(n, len(all_files)))
    print(f"共 {len(all_files)} 个文件，抽取 {len(sampled)} 个")

    for src in sampled:
        rel = os.path.relpath(src, src_dir)
        dst = os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    print(f"完成，输出至: {dst_dir}")

if __name__ == "__main__":

    n = 1000
    # src = "/media/marx/My Passport/camera"
    src = "/media/marx/My Passport/Collection"
    dst = os.path.join(src + "_" + str(n))



    sample_files(src, dst, n)