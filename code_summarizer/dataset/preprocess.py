# import os
# import torch
# import h5py
# import sentencepiece as spm  # BPE tokenizer

# # File paths
# decl_path = "parallel_decl.txt"
# body_path = "parallel_bodies.txt"
# desc_path = "parallel_desc"
# output_h5 = "processed_data.h5"

# # 1️⃣ Read files
# with open(decl_path, "r", encoding="utf-8") as f:
#     decl_text = f.read().strip()
# with open(body_path, "r", encoding="utf-8") as f:
#     body_text = f.read().strip()
# with open(desc_path, "r", encoding="ISO-8859-1") as f:
#     desc_text = f.read().strip()

# # 2️⃣ Split into functions using `\n\n` separator
# decl_data = decl_text.split("\n\n")
# body_data = body_text.split("\n\n")
# desc_data = desc_text.split("\n\n")

# # Ensure alignment
# assert len(decl_data) == len(body_data) == len(desc_data), "Data misalignment detected!"

# # 3️⃣ Concatenate `decl + body` for each function
# full_functions = [decl + " " + body for decl, body in zip(decl_data, body_data)]

# # 4️⃣ Train BPE tokenizer (if not already trained)
# bpe_model = "bpe_model.model"
# if not os.path.exists(bpe_model):
#     with open("training_text.txt", "w", encoding="utf-8") as f:
#         f.write("\n".join(full_functions + desc_data))  # Train on both code & descriptions
#     spm.SentencePieceTrainer.Train(f"--input=training_text.txt --model_prefix=bpe_model --vocab_size=8000")

# # Load trained BPE model
# sp = spm.SentencePieceProcessor(model_file="bpe_model.model")

# # 5️⃣ Encode the data
# X_encoded = [sp.encode_as_ids(func) for func in full_functions]
# Y_encoded = [sp.encode_as_ids(desc) for desc in desc_data]

# # Convert to PyTorch tensors
# X_tensor = [torch.tensor(seq, dtype=torch.long) for seq in X_encoded]
# Y_tensor = [torch.tensor(seq, dtype=torch.long) for seq in Y_encoded]

# print(X_tensor.shape)
# print(Y_tensor.shape)

# # 6️⃣ Save tensors to `.h5`
# with h5py.File(output_h5, "w") as h5f:
#     h5f.create_dataset("X", data=torch.nn.utils.rnn.pad_sequence(X_tensor, batch_first=True).numpy())
#     h5f.create_dataset("Y", data=torch.nn.utils.rnn.pad_sequence(Y_tensor, batch_first=True).numpy())

import os
import torch
import h5py
import sentencepiece as spm  # BPE tokenizer

# File paths
decl_path = "parallel_decl.txt"
body_path = "parallel_bodies.txt"
desc_path = "parallel_desc"
output_h5 = "processed_data.h5"

# 1️⃣ Read files
with open(decl_path, "r", encoding="utf-8") as f:
    decl_text = f.read().strip()
with open(body_path, "r", encoding="utf-8") as f:
    body_text = f.read().strip()
with open(desc_path, "r", encoding="ISO-8859-1") as f:
    desc_text = f.read().strip()

# 2️⃣ Split into functions using `\n\n` separator
decl_data = decl_text.split("\n")
print(len(decl_data))
body_data = body_text.split("\n")
print(len(body_data))
desc_data = desc_text.split("\n")
print(len(desc_data))

MAX_SAMPLES = 500
decl_data = decl_data[:MAX_SAMPLES]
body_data = body_data[:MAX_SAMPLES]
desc_data = desc_data[:MAX_SAMPLES]

# Ensure alignment
assert len(decl_data) == len(body_data) == len(desc_data), "Data misalignment detected!"

# 3️⃣ Concatenate `decl + body` for each function
full_functions = [decl + " " + body for decl, body in zip(decl_data, body_data)]

# 4️⃣ Train BPE tokenizer (if not already trained)
bpe_model = "bpe_model.model"
if not os.path.exists(bpe_model):
    with open("training_text.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(full_functions + desc_data))  # Train on both code & descriptions
    spm.SentencePieceTrainer.Train(f"--input=training_text.txt --model_prefix=bpe_model --vocab_size=8000")

# Load trained BPE model
sp = spm.SentencePieceProcessor(model_file="bpe_model.model")

# 5️⃣ Encode the data
X_encoded = [sp.encode_as_ids(func) for func in full_functions]
Y_encoded = [sp.encode_as_ids(desc) for desc in desc_data]

# Debugging check: Make sure encoded values are integers
# Print the first 5 examples from X_encoded and Y_encoded along with their lengths
for i in range(5):
    print(f"X[{i}] (len={len(X_encoded[i])}): {X_encoded[i]}")
    print(f"Y[{i}] (len={len(Y_encoded[i])}): {Y_encoded[i]}")
    print("-" * 50)

# Convert to PyTorch tensors
X_tensor = [torch.tensor(seq, dtype=torch.long) for seq in X_encoded]
Y_tensor = [torch.tensor(seq, dtype=torch.long) for seq in Y_encoded]

# 6️⃣ Padding sequences (Use 0 as padding)
X_padded = torch.nn.utils.rnn.pad_sequence(X_tensor, batch_first=True, padding_value=0)
Y_padded = torch.nn.utils.rnn.pad_sequence(Y_tensor, batch_first=True, padding_value=0)

# 7️⃣ Store tensors in HDF5 (Ensure int64 storage)
with h5py.File(output_h5, "w") as h5f:
    h5f.create_dataset("X", data=X_padded.numpy().astype("int64"))
    h5f.create_dataset("Y", data=Y_padded.numpy().astype("int64"))

print("✅ Preprocessing complete! Data saved in 'processed_data.h5'.")
