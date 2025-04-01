# import torch
# from torch.utils.data import Dataset, DataLoader

# class CodeSummarizationDataset(Dataset):
#     def __init__(self, source_texts, target_texts, source_vocab, target_vocab):
#         self.source_texts = source_texts
#         self.target_texts = target_texts
#         self.source_vocab = source_vocab
#         self.target_vocab = target_vocab

#     def __len__(self):
#         return len(self.source_texts)

#     def __getitem__(self, idx):
#         source_tokens = [self.source_vocab.get(word, 1) for word in self.source_texts[idx].split()]
#         target_tokens = [self.target_vocab.get(word, 1) for word in self.target_texts[idx].split()]
#         return torch.tensor(source_tokens), torch.tensor(target_tokens)

# def get_dataloader(source_texts, target_texts, source_vocab, target_vocab, batch_size=32, shuffle=True):
#     dataset = CodeSummarizationDataset(source_texts, target_texts, source_vocab, target_vocab)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
