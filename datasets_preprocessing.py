import os
import json

def remove_empty_speaker_positions(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    novel_statistics = {}
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            filtered_data = [entry for entry in data if entry['speaker_position']]

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=4)
            novel_statistics[filename] = len(filtered_data)

    return novel_statistics

input_dir = "C:/Users/Lenovo/OneDrive/NUS/CS-24fall/project/AudiobookGeneration_cs5647/LiteraryTextsDataset/SID"
output_dir = "C:/Users/Lenovo/OneDrive/NUS/CS-24fall/project/AudiobookGeneration_cs5647/LiteraryTextsDataset/SID"
novel_statistics = remove_empty_speaker_positions(input_dir, output_dir)
def save_statistics_to_file(statistics, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=4)

statistics_output_file = "C:/Users/Lenovo/OneDrive/NUS/CS-24fall/project/AudiobookGeneration_cs5647/LiteraryTextsDataset/novel_statistics.json"
novel_statistics = remove_empty_speaker_positions(input_dir, output_dir)
save_statistics_to_file(novel_statistics, statistics_output_file)
print(f"Novel statistics saved to: {statistics_output_file}")