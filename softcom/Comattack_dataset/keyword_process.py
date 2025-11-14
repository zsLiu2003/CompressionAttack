import json
import re

def extract_demo_blocks(text):
    """
    Extract multiple 'Modified Demo' blocks from a single demo field.
    """
    pattern = r'"Modified Demo":.*?(?=(?:"Modified Demo":|$))'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return [f'\n\n{match.strip()}' for match in matches]

def extract_product_name(text):
    """
    Extract product name from a 'Modified Demo' block using common delimiters.
    """
    match = re.search(r'Modified Demo"\s*:\s*(.*?)(?:\s*[\u2013:-])', text)
    if match:
        return match.group(1).strip()
    return None

def remove_start_with_quote(text):
    """
    Remove leading quotes from the product name if present.
    """
    if text.startswith('"'):
        return text[1:].strip()
    return text.strip()

def filter_and_split_each_block(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = []

    # Process each block independently
    for group in data:
        seen_products = set()
        split_demos = []

        # Iterate over demo_1 ... demo_5 in current block
        for key in sorted(group.keys()):
            demo_text = group[key]
            demo_blocks = extract_demo_blocks(demo_text)
            for block in demo_blocks:
                product_name = extract_product_name(block)

                if product_name and product_name not in seen_products and remove_start_with_quote(product_name) not in seen_products and "\"" + product_name not in seen_products:
                    # print(f"Processing product: {product_name}")
                    # print("\"" + product_name)
                    seen_products.add(product_name)
                    split_demos.append(block)

        new_group = {}
        for i, demo in enumerate(split_demos):
            new_group[f'demo_{i+1}'] = demo

        new_data.append(new_group)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(new_data)} unique demos to {output_path}")

filter_and_split_each_block('keywords_decrease.json', 'keywords_decrease_filtered.json')
filter_and_split_each_block('keywords_increase.json', 'keywords_increase_filtered.json')

