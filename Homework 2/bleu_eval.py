import sys
import json
from nltk.translate.bleu_score import sentence_bleu

def load_test_data(test_file):
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    return test_data

def evaluate_bleu(output_file, test_data):
    result = {}
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            test_id = line[0]
            caption = line[1]
            result[test_id] = caption

    bleu_scores = []
    for item in test_data:
        captions = [ref.rstrip('.') for ref in item['caption']]
        try:
            bleu_score = sentence_bleu(captions, result[item['id']].rstrip('.'))
            bleu_scores.append(bleu_score)
        except KeyError as e:
            print(f"Warning: No caption found for video ID '{item['id']}'")

    return bleu_scores

def main(output_file, test_file):
    test_data = load_test_data(test_file)
    bleu_scores = evaluate_bleu(output_file, test_data)
    
    if bleu_scores:
        average_bleu = sum(bleu_scores) / len(bleu_scores)
        print("Average BLEU score is:", average_bleu)
    else:
        print("No BLEU scores calculated.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_bleu.py <output_file> <test_file>")
        sys.exit(1)

    output_file = sys.argv[1]
    test_file = sys.argv[2]
    
    main(output_file, test_file)
