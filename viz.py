# load models from disk and make predictions
from background_and_models import *
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import Counter

base_path = './models/best_baseDistilBERT_CE.pth'
aug_path = './models/best_augDistilBERT_CE.pth'

base_gram_path = './models/gramdistilBERT_baseCE.pth'
aug_gram_path = './models/gramdistilBERT_augCE.pth'

# load the test data
matched_path = "./multinli_1.0/multinli_1.0_dev_matched.jsonl"

# output attentions so we can analyze later
distilbert_model = AutoModel.from_pretrained("distilbert-base-uncased", output_attentions=True)
distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# full models
base_model = DistilBERTClassifier(distilbert_model, distilbert_tokenizer).to(Config.device)
aug_model = DistilBERTClassifier(distilbert_model, distilbert_tokenizer).to(Config.device)

# grammatical ones
gram_base_model = DistilBERTClassifier(distilbert_model, distilbert_tokenizer).to(Config.device)
gram_aug_model = DistilBERTClassifier(distilbert_model, distilbert_tokenizer).to(Config.device)


base_model_state_dict = torch.load(base_path, map_location=Config.device)
base_model.load_state_dict(base_model_state_dict)

aug_model_state_dict = torch.load(aug_path, map_location=Config.device)
aug_model.load_state_dict(aug_model_state_dict)

gram_base_model_state_dict = torch.load(base_gram_path, map_location=Config.device)
gram_base_model.load_state_dict(gram_base_model_state_dict)

gram_aug_model_state_dict = torch.load(aug_gram_path, map_location=Config.device)
gram_aug_model.load_state_dict(gram_aug_model_state_dict)

matched_data = load_data(matched_path)

# get the predictions for each model
labels = ['entailment', 'contradiction', 'neutral']
def test_wrong_pred(test_data, classifier):
    all_outputs = []
    all_targets = []
    random.shuffle(test_data)
    with torch.no_grad():
        for datum in tqdm(test_data):
            sent1 = datum.get_sent1()
            sent2 = datum.get_sent2()
            classifier_outputs = classifier(sent1, sent2)
            classifier_outputs = classifier_outputs.cpu().numpy()
            outputs = np.argmax(classifier_outputs, axis=1)
            targets = [labels.index(datum.get_gold_label())]
            all_outputs.extend(outputs)
            all_targets.extend(targets)
    
    print(classification_report(all_targets, all_outputs, target_names=labels))
    cm = confusion_matrix(all_targets, all_outputs)
    
    # generate examples where the model predicts the wrong label and correct label
    wrong_predictions = [(test_data[i].get_sent1(), test_data[i].get_sent2(), labels[all_outputs[i]], labels[all_targets[i]]) for i in range(len(all_outputs)) if all_outputs[i] != all_targets[i]]
    correct_predictions = [(test_data[i].get_sent1(), test_data[i].get_sent2(), labels[all_outputs[i]], labels[all_targets[i]]) for i in range(len(all_outputs)) if all_outputs[i] == all_targets[i]]
    
    print(f"Wrong predictions: {len(wrong_predictions)}")
    print(f"Correct predictions: {len(correct_predictions)}")
    print(f'Confusion matrix: {cm}\n')

    # adding this to add to list
    return wrong_predictions, correct_predictions

# write the predictions to a file in a json readable format
def write_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        json.dump(predictions, f, indent=4)

# get the wrong predictions for each model
print("Base Model:")
base_wrong_preds, base_correct_preds = test_wrong_pred(matched_data, base_model)
print("\nAug Model:")
aug_wrong_preds, aug_correct_preds = test_wrong_pred(matched_data, aug_model)
print("\nGram base model:")
gram_base_wrong_preds, gram_base_correct_preds = test_wrong_pred(matched_data, gram_base_model)
print("\nGram aug model:")
gram_aug_wrong_preds, gram_aug_correct_preds = test_wrong_pred(matched_data, gram_aug_model)

# write all predictions to a file
write_predictions_to_file(base_wrong_preds, './outputs/base_wrong_preds.json')
write_predictions_to_file(base_correct_preds, './outputs/base_correct_preds.json')
write_predictions_to_file(aug_wrong_preds, './outputs/aug_wrong_preds.json')
write_predictions_to_file(aug_correct_preds, './outputs/aug_correct_preds.json')
write_predictions_to_file(gram_base_wrong_preds, './outputs/gram_base_wrong_preds.json')
write_predictions_to_file(gram_base_correct_preds, './outputs/gram_base_correct_preds.json')
write_predictions_to_file(gram_aug_wrong_preds, './outputs/gram_aug_wrong_preds.json')
write_predictions_to_file(gram_aug_correct_preds, './outputs/gram_aug_correct_preds.json')