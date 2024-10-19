import xml.etree.ElementTree as ET
import json
import os

def get_paragraph_range(paragraphs, target_idx, window_size=300):
    def count_words_in_paragraph(p):
        return len(' '.join(p.itertext()).split())
    num_paragraphs = len(paragraphs)
    half_window = window_size // 2
    start_idx, end_idx = target_idx, target_idx

    total_words = count_words_in_paragraph(paragraphs[target_idx]) // 2
    while start_idx > 0 and total_words < half_window:
        start_idx -= 1
        total_words += count_words_in_paragraph(paragraphs[start_idx])

    total_words = count_words_in_paragraph(paragraphs[target_idx]) // 2
    while end_idx < num_paragraphs - 1 and total_words < half_window:
        end_idx += 1
        total_words += count_words_in_paragraph(paragraphs[end_idx])

    return start_idx, end_idx


def extract_text_with_spaces(element):
    parts = []
    for part in element.itertext():
        parts.append(part.strip())
    return ' '.join(parts)

def extract_sid_data_with_window(xml_file, window_size):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    paragraphs = [p for p in root.findall('.//tei:p', ns) if 'n' in p.attrib]
    dataset = []

    for i, p in enumerate(paragraphs):
        for said in p.findall('.//tei:said', ns):
            dialogue = extract_text_with_spaces(said)
            speaker_label = said.get('who')

            start_idx, end_idx = get_paragraph_range(paragraphs, i, window_size)
            context = ' '.join(
                [extract_text_with_spaces(paragraphs[j]) for j in range(start_idx, end_idx + 1)]
            )

            # 添加到数据集中
            dataset.append({
                'context': context,
                'target_dialogue_text': dialogue,
                'speaker_label': speaker_label
            })
    return dataset

def extract_character_aliases(xml_file):
    """
    提取 XML 文件中的人物及其所有名字变体，返回 {id: [names]} 的映射。
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}  # 命名空间定义

    character_aliases = {}

    for person in root.findall('.//tei:person', ns):
        person_id = person.get('{http://www.w3.org/XML/1998/namespace}id')
        names = []
        for pers_name in person.findall('.//tei:persName', ns):
            if pers_name.text:
                names.append(pers_name.text.strip())
            for add_name in pers_name.findall('.//tei:addName', ns):
                if add_name.text:
                    names.append(add_name.text.strip())

        if person_id and names:
            character_aliases[person_id] = names

    return character_aliases




def get_speaker_positions(context, speaker_name):
    words = context.split()
    speaker_words = speaker_name.split()
    speaker_len = len(speaker_words)
    positions = []

    for i in range(len(words) - speaker_len + 1):
        if words[i:i + speaker_len] == speaker_words:
            positions.append([i, i + speaker_len - 1])

    return positions

def convert_dataset_with_window(xml_file, window_size=15):
    character_aliases = extract_character_aliases(xml_file)
    raw_data = extract_sid_data_with_window(xml_file, window_size)
    formatted_data = []

    for entry in raw_data:
        context = entry['context']
        target_dialogue = entry['target_dialogue_text']
        speaker_label = entry['speaker_label']

        if speaker_label is None:
            print("Warning: Missing 'who' attribute in <said> tag. Skipping this entry.")
            continue

        speaker_name_list = character_aliases.get(speaker_label.lstrip('#'), [])
        speaker_positions = []
        for speaker_name in speaker_name_list:
            speaker_positions.extend(get_speaker_positions(context, speaker_name))

        target_coords = get_speaker_positions(context, target_dialogue)

        formatted_data.append({
            'context': context,
            'target_dialogue': target_coords[0] if target_coords else [0, 0],
            'speaker_position': speaker_positions
        })

    return formatted_data

def save_to_json(data, output_dir, xml_file):
    file_name = os.path.basename(xml_file).replace('.xml', '.json')
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    input_dir = "C:/Users/Lenovo/OneDrive/NUS/CS-24fall/project/AudiobookGeneration_cs5647/LiteraryTextsDataset/data"
    output_dir = "C:/Users/Lenovo/OneDrive/NUS/CS-24fall/project/AudiobookGeneration_cs5647/LiteraryTextsDataset/SID"

    # input_path = "C:/Users/Lenovo/OneDrive/NUS/CS-24fall/project/AudiobookGeneration_cs5647/LiteraryTextsDataset/data/AnneofGreenGablesbyLMLucyMaudMontgomery45.xml"
    # data = convert_dataset_with_window(input_path, 500)
    # save_to_json(data, output_dir, input_path)

    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".xml"):
            xml_file = os.path.join(input_dir, filename)
            print(f"Processing: {xml_file}")
            data = convert_dataset_with_window(xml_file, 500)
            save_to_json(data, output_dir, xml_file)

    print("All files processed successfully.")
