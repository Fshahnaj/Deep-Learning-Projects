import sys
import torch
import json
from nltk.translate.bleu_score import sentence_bleu
from SeqtoSeqTrain import test_data, test, MODELS, encoderRNN, decoderRNN, attention
from torch.utils.data import DataLoader
from bleu_eval import evaluate_bleu # Assuming calculate_bleu is the correct function name
import pickle

# Load the model
model = torch.load('/content/gdrive/MyDrive/hw2_hw2_1/SavedModel/model0.pth', map_location=lambda storage, loc: storage)
filepath = '/content/gdrive/MyDrive/hw2_hw2_1/testing_data/feat'

# Load the dataset
dataset = test_data('{}'.format(sys.argv[1]))
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

# Load the word index mapping
with open('/content/gdrive/MyDrive/hw2_hw2_1/i2w.pickle', 'rb') as handle:
    i2w = pickle.load(handle)

# Move the model to GPU if available
model = model.cuda()

# Test the model
ss = test(testing_loader, model, i2w)

# Save the results
output_file = sys.argv[2]
with open(output_file, 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))

# Load the testing labels
test = json.load(open('/content/gdrive/MyDrive/hw2_hw2_1/testing_data/testing_label.json'))

# Calculate BLEU scores
result = {}
with open(output_file, 'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption

# Calculate BLEU scores for each item
bleu_scores = []
for item in test:
    reference_captions = [x.rstrip('.') for x in item['caption']]
    candidate_caption = result[item['id']]
    bleu_score = calculate_bleu(reference_captions, candidate_caption) # Assuming calculate_bleu takes care of the BLEU calculation
    bleu_scores.append(bleu_score)

# Calculate the average BLEU score
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print("Average BLEU score is", average_bleu_score)
