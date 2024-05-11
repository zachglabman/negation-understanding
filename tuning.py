from background_and_models import *
import gc

train_path = "./multinli_1.0/multinli_1.0_train.jsonl"
matched_path = "./multinli_1.0/multinli_1.0_dev_matched.jsonl"
augmented_path = "./multinli_1.0/augmented_dataset.json"

train_data = load_data(train_path)
dev_matched = load_data(matched_path)

# load augmented data if it exists
if os.path.exists(augmented_path):
    with open(augmented_path, "r") as json_file:
        loaded_data = json.load(json_file)
    augmented_train = [Datum.from_dict(data) for data in loaded_data]

else:
    neg_train_data = create_negated(train_data)
    random.shuffle(neg_train_data)
    augmented_train = train_data + neg_train_data
    
    serialized_data = [datum.to_dict() for datum in augmented_train]
    
    with open(augmented_path, "w") as json_file:
        json.dump(serialized_data, json_file)

f1s_matched = {}

num_epochs = [3, 4, 5]
lr = [1e-5, 5e-5]
batch_size = [32, 64, 128]

best_f1_base = 0
best_f1_aug = 0
best_f1_overall = 0
best_model_name = ""

# random search
for i in range(25):
    epoch = random.choice(num_epochs)
    # random choice of any number between 1e-5 and 5e-5
    l = random.uniform(lr[0], lr[1])
    b = random.choice(batch_size)
    
    print("\n\n\n---------------------------------\n\n\n")
    print(f"\n\n\nRandom Tuning Episode {i}:\n\nEpochs: {epoch}, Learning Rate: {l}, Batch Size: {b}\n")

    classifier_base, classifier_aug, optimizer_base, optimizer_aug = initialize_DB_models(lr=l)
    criterion = "cross_entropy"

    # Fine-tune base model
    model_name = "baseDistilBERT_CE"
    mn = f'{model_name}_{epoch}_{l}_{b}'
    print(f"\n{model_name}_{epoch}_{l}_{b} Fine-tuned with last layer unfrozen:\n")
    train_classifier_BERT(train_data, classifier_base, optimizer_base, criterion, epoch, b)
    f1s_matched[mn] = test_classifier_BERT(dev_matched, classifier_base)

    # Save best model
    if f1s_matched[mn] > best_f1_base:
        best_f1_base = f1s_matched[mn]
        torch.save(classifier_base.state_dict(), f"./models/best_{model_name}.pth")
    
    # Fine-tune augmented model
    classifier_base, classifier_aug, optimizer_base, optimizer_aug = initialize_DB_models(lr=l)
    model_name = "augDistilBERT_CE"
    mn = f'{model_name}_{epoch}_{l}_{b}'
    print(f"\n{mn} Fine-tuned with last layer unfrozen:\n")
    train_classifier_BERT(augmented_train, classifier_aug, optimizer_aug, criterion, epoch, b)
    f1s_matched[mn] = test_classifier_BERT(dev_matched, classifier_aug)

    # Save best model
    if f1s_matched[mn] > best_f1_aug:
        best_f1_aug = f1s_matched[mn]
        torch.save(classifier_aug.state_dict(), f"./models/best_{model_name}.pth")
    
    # sort by f1 score descending
    f1s_matched = dict(sorted(f1s_matched.items(), key=lambda item: item[1], reverse=True))
    # print the model and highest f1
    print(f"Best model: {list(f1s_matched.keys())[0]} with F1: {list(f1s_matched.values())[0]}\n\n\n")

    # call garbage collector
    gc.collect()
        

print("\n\n\n---------------------------------\n\n\n")
for key in f1s_matched.keys():
    print(f"F1 score for {key} on matched dev set: {f1s_matched[key]}")
    