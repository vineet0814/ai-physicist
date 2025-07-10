import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import config 

class JSONDataset(Dataset):
    """
    A PyTorch Dataset for loading JSON data.

    Args:
        data (list): A list of dictionaries containing the dataset.
        tokenizer: The tokenizer to tokenize the data.
        max_length (int): Maximum token length for each input.
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item.get("instruction", "") + " " + item.get("input", "")
        output_text = item.get("output", "")

        # Tokenize input and output
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        outputs = self.tokenizer(
            output_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": outputs["input_ids"].squeeze(0)
        }

def load_and_split_dataset(json_file, tokenizer, test_fraction=0.1, eval_fraction=0.1, max_length=512):
    """
    Load a JSON dataset and split it into train, test, and eval sets.

    Args:
        json_file (str): Path to the JSON dataset file.
        tokenizer: The tokenizer to tokenize the data.
        test_fraction (float): Fraction of the dataset to use for the test set.
        eval_fraction (float): Fraction of the dataset to use for the eval set.
        max_length (int): Maximum token length for each input.

    Returns:
        train_dataset: Tokenized training dataset.
        test_dataset: Tokenized test dataset.
        eval_dataset: Tokenized evaluation dataset.
    """
    # Load the dataset
    with open(json_file, "r") as f:
        data = [json.loads(line) for line in f]

    # Split the dataset into train, test, and eval sets
    train_data, test_data = train_test_split(data, test_size=test_fraction, random_state=42)
    train_data, eval_data = train_test_split(train_data, test_size=eval_fraction / (1 - test_fraction), random_state=42)

    # Create tokenized datasets
    train_dataset = JSONDataset(train_data, tokenizer, max_length=max_length)
    test_dataset = JSONDataset(test_data, tokenizer, max_length=max_length)
    eval_dataset = JSONDataset(eval_data, tokenizer, max_length=max_length)

    return train_dataset, test_dataset, eval_dataset