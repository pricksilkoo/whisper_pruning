import os
# ⚠️ 必须放在 import datasets 之前！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

data_dir = "../data/fleurs_full"
os.makedirs(data_dir, exist_ok=True)

languages = {
    "cmn_hans_cn": "zh",
    "en_us": "en",
    "ja_jp": "jp",
    "ko_kr": "kr",
    "fr_fr": "fr",
    "de_de": "de",
    "es_419": "es",
    "ru_ru": "ru",
    "hi_in": "hi",
    "ar_eg": "ar"
}

def download_and_save(lang_code, folder_name):
    save_path = os.path.join(data_dir, folder_name)
    
    if os.path.exists(save_path):
        print(f"--- {folder_name} ({lang_code}) 已存在，跳过 ---")
        return

    # 注意看这句打印，如果终端出来的不是这句，说明你没保存！
    print(f"正在通过镜像源高速下载 {lang_code} ...")
    try:
        dataset = load_dataset("google/fleurs", lang_code, trust_remote_code=True)
        dataset.save_to_disk(save_path)
        print(f"✅ Successfully saved {lang_code} to {save_path}")
    except Exception as e:
        print(f"❌ Failed to download {lang_code}: {e}")

for lang_code, folder_name in languages.items():
    download_and_save(lang_code, folder_name)

print("\n🎉 所有选定语言已处理完毕！")