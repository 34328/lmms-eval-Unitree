#!/usr/bin/env python3
"""
Convert MetaVQA JSON from dict-of-records format to a list-of-records format
and remove the `options` field from each record.

Usage:
  python convert_test_json.py \
      --input MetaVQA/test.json \
      --output MetaVQA/test_fixed.json

This script is idempotent: if input is already a list, it will still remove any
`options` fields and rewrite to the output path.
"""
import argparse
import json
import os
import shutil


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def convert(data):
    # If top-level is a dict, convert to list of values (preserve insertion order)
    if isinstance(data, dict):
        records = list(data.values())
    elif isinstance(data, list):
        records = data
    else:
        raise TypeError(f'Unsupported top-level JSON type: {type(data)}')

    # Remove `options` key from each record if present
    removed_count = 0
    for rec in records:
        if isinstance(rec, dict) and 'options' in rec:
            rec.pop('options', None)
            removed_count += 1
    return records, removed_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='MetaVQA/test.json', help='input JSON path')
    parser.add_argument('--output', '-o', default='MetaVQA/test_fixed.json', help='output JSON path')
    parser.add_argument('--backup', action='store_true', help='backup existing output file if present')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        raise SystemExit(f'Input file not found: {input_path}')

    data = load_json(input_path)

    records, removed = convert(data)

    # Optional backup
    if args.backup and os.path.exists(output_path):
        bak = output_path + '.bak'
        print(f'Backing up existing output to {bak}')
        shutil.copy2(output_path, bak)

    write_json(output_path, records)

    print(f'Wrote {len(records)} records to {output_path}')
    print(f'Removed `options` from {removed} records')


if __name__ == '__main__':
    main()
