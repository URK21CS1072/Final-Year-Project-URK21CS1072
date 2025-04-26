# ==============================
# Dysarthria Classification Pipeline
# ==============================

# ------------------------------
# 1. Imports and Setup
# ------------------------------
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import librosa
import librosa.display
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch.nn.utils.prune as prune

# ------------------------------
# 2. Data Collection and Preprocessing
# ------------------------------

def preprocess_transcript(text):
    """
    Replace non-word annotations with <NONWORD> token.
    Example: "[say Ah-P-Eee repeatedly]" -> "<NONWORD>"
    """
    return re.sub(r'\[.*?\]', '<NONWORD>', text)

def collect_data(base_dir):
    """
    Traverse through the dataset directory and collect audio and transcript paths.
    Only process directories that correspond to sessions (e.g., session1, session2_3).
    Handle both 'prompts' and 'promps' transcript directories.
    """
    data = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Base directory {base_dir} does not exist.")
        return data
    
    speakers = [speaker for speaker in base_path.iterdir() if speaker.is_dir()]
    
    for speaker in speakers:
        print(f"Processing speaker: {speaker.name}")
        # Identify session directories (e.g., session1, session2_3)
        sessions = [session for session in speaker.iterdir() if session.is_dir() and session.name.lower().startswith('session')]
        if not sessions:
            print(f"  No session directories found for speaker {speaker.name}. Skipping.")
            continue
        
        for session in sessions:
            print(f"  Processing session: {session.name}")
            wav_arraymic_dir = session / 'wav_arraymic'
            wav_headmic_dir = session / 'wav_headmic'
            prompts_dir = session / 'prompts'
            promps_dir = session / 'promps'

            # Handle both 'prompts' and 'promps' directories
            transcript_dirs = []
            if prompts_dir.exists():
                transcript_dirs.append(prompts_dir)
                print(f"    Found prompts directory: {prompts_dir}")
            if promps_dir.exists():
                transcript_dirs.append(promps_dir)
                print(f"    Found promps directory: {promps_dir}")
            if not transcript_dirs:
                print(f"    No prompts or promps directory found in {session}. Skipping session.")
                continue  # Skip if no transcript directory is found

            # Collect audio files from both mic directories
            for mic_dir in [wav_arraymic_dir, wav_headmic_dir]:
                if mic_dir.exists():
                    print(f"    Processing microphone directory: {mic_dir}")
                    for wav_file in mic_dir.glob('*.wav'):
                        print(f"      Found audio file: {wav_file.name}")
                        transcript_file = None
                        for t_dir in transcript_dirs:
                            potential_transcript = t_dir / (wav_file.stem + '.txt')
                            if potential_transcript.exists():
                                transcript_file = potential_transcript
                                break
                        if transcript_file:
                            print(f"        Found transcript: {transcript_file.name}")
                            try:
                                with open(transcript_file, 'r', encoding='utf-8') as f:
                                    transcript = f.read().strip()
                                preprocessed_transcript = preprocess_transcript(transcript)
                                # Assign label based on base directory name
                                label = 1 if 'with dysarthria' in base_dir.lower() else 0
                                data.append({
                                    'audio': str(wav_file.resolve()),
                                    'transcript': preprocessed_transcript,
                                    'label': label
                                })
                            except Exception as e:
                                print(f"        Error reading {transcript_file.name}: {e}")
                        else:
                            print(f"        Transcript for {wav_file.name} not found.")
                else:
                    print(f"    Microphone directory not found: {mic_dir}")
    return data

def prepare_dataset(base_dirs):
    """
    Collect data from multiple base directories (e.g., dysarthric and non_dysarthric)
    and compile them into a single DataFrame.
    """
    all_data = []
    for base_dir in base_dirs:
        data = collect_data(base_dir)
        all_data.extend(data)
    if all_data:
        df = pd.DataFrame(all_data)
        print(f"Total samples collected: {len(df)}")
        print(df.head())
        return df
    else:
        print("No data collected. Please verify the dataset structure and file naming.")
        return pd.DataFrame()

# Define base directories for dysarthric and non_dysarthric data
dysarthric_dir = r'D:\datasets\TORGO male with dysarthria'       # Adjust as needed
non_dysarthric_dir = r'D:\datasets\TORGO male without dysarthria'  # Adjust as needed

# Prepare the dataset
df = prepare_dataset([dysarthric_dir, non_dysarthric_dir])

# ------------------------------
# 3. Dataset Class with On-the-Fly Feature Extraction
# ------------------------------

class DysarthriaDataset(Dataset):
    def __init__(self, dataframe, tokenizer, transform=None, 
                 max_spectrogram_time=200, max_mfcc_time=200, n_fft=512):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'audio', 'transcript', and 'label' columns.
            tokenizer: Tokenizer for textual data.
            transform: Transformations for Mel-spectrogram images (optional).
            max_spectrogram_time (int): Fixed number of time steps for spectrograms.
            max_mfcc_time (int): Fixed number of time steps for MFCCs.
            n_fft (int): FFT window size.
        """
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_spectrogram_time = max_spectrogram_time
        self.max_mfcc_time = max_mfcc_time
        self.n_fft = n_fft

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        audio_path = sample['audio']
        transcript = sample['transcript']
        label = sample['label']

        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=22050)  # Ensure consistent sampling rate
        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Pad audio if it's shorter than n_fft
        if len(audio) < self.n_fft:
            pad_length = self.n_fft - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')

        # Compute Mel-spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000, n_fft=self.n_fft)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram_db = (spectrogram_db - spectrogram_db.mean()) / (spectrogram_db.std() + 1e-9)
        spectrogram_db = librosa.util.fix_length(spectrogram_db, size=self.max_spectrogram_time, axis=1)
        spectrogram_db = np.expand_dims(spectrogram_db, axis=0)  # [1, n_mels, time_steps]
        spectrogram_tensor = torch.tensor(spectrogram_db, dtype=torch.float)

        # Compute MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=self.n_fft)
        mfcc = librosa.util.fix_length(mfcc, size=self.max_mfcc_time, axis=1)
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
        mfcc_tensor = torch.tensor(mfcc.T, dtype=torch.float)  # [time_steps, 40]

        # Load Articulatory Features (if available)
        # If articulatory features are not available, you can set them to zeros or omit them
        articulatory = np.zeros(10, dtype=np.float32)  # Example: 10 articulatory features
        articulatory_tensor = torch.tensor(articulatory, dtype=torch.float)

        # Tokenize transcript
        tokens = self.tokenizer(
            transcript,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )
        input_ids = tokens['input_ids'].squeeze(0)  # [128]
        attention_mask = tokens['attention_mask'].squeeze(0)  # [128]

        return {
            'spectrogram': spectrogram_tensor,      # [1, 128, max_spectrogram_time]
            'mfcc': mfcc_tensor,                    # [max_mfcc_time, 40]
            'articulatory': articulatory_tensor,    # [10]
            'input_ids': input_ids,                 # [128]
            'attention_mask': attention_mask,       # [128]
            'label': torch.tensor(label, dtype=torch.long)  # [1]
        }

# ------------------------------
# 4. Feature Extraction Modules
# ------------------------------

# a. CNN for Mel-Spectrograms
class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=128):
        super(SpectrogramCNN, self).__init__()
        self.features = nn.Sequential(
            # Convolution Block 1
            nn.Conv2d(1, 24, kernel_size=3, padding=1, groups=1),  # Changed to groups=1 for consistency
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=1),  # Changed to groups=1
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution Block 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1, groups=1),  # Changed to groups=1
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x  # [batch, num_classes]

# b. RNN for MFCCs
class MFCCRNN(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, output_dim=128):
        super(MFCCRNN, self).__init__()
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x: [batch, time_steps, input_dim]
        output, _ = self.gru(x)
        # Mean pooling over time steps
        output = output.mean(dim=1)
        output = self.fc(output)
        return output  # [batch, output_dim]

# c. Transformer Encoder for Textual Data
class TextTransformer(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', output_dim=128):
        super(TextTransformer, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        cls_output = self.fc(cls_output)
        return cls_output  # [batch, output_dim]

# ------------------------------
# 5. Encoder Class
# ------------------------------
class Encoder(nn.Module):
    def __init__(self, spectrogram_cnn, mfcc_rnn, text_transformer):
        super(Encoder, self).__init__()
        self.spectrogram_cnn = spectrogram_cnn
        self.mfcc_rnn = mfcc_rnn
        self.text_transformer = text_transformer

    def forward(self, spectrogram, mfcc, articulatory, input_ids, attention_mask):
        spectrogram_feat = self.spectrogram_cnn(spectrogram)  # [batch, 128]
        mfcc_feat = self.mfcc_rnn(mfcc)                      # [batch, 128]
        text_feat = self.text_transformer(input_ids, attention_mask)  # [batch, 128]
        combined = torch.cat((spectrogram_feat, mfcc_feat, text_feat), dim=1)  # [batch, 384]
        return combined

# ------------------------------
# 6. Complete Model Architectures
# ------------------------------

# a. Dysarthria Classifier
class DysarthriaClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super(DysarthriaClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),  # 384 = 128 (spectrogram) + 128 (MFCC) + 128 (text)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, spectrogram, mfcc, articulatory, input_ids, attention_mask):
        combined = self.encoder(spectrogram, mfcc, articulatory, input_ids, attention_mask)  # [batch, 384]
        out = self.classifier(combined)  # [batch, num_classes]
        return out

# b. SimCLR Pretrainer
class SimCLRPretrainer(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super(SimCLRPretrainer, self).__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(384, 256),  # Input features from encoder
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, spectrogram, mfcc, articulatory, input_ids, attention_mask):
        features = self.encoder(spectrogram, mfcc, articulatory, input_ids, attention_mask)  # [batch, 384]
        projections = self.projection_head(features)  # [batch, 128]
        return projections

# ------------------------------
# 7. Self-Supervised Pretraining
# ------------------------------

# Contrastive Loss Function
def contrastive_loss(x, temperature=0.5):
    """
    Computes the contrastive loss as defined in SimCLR.
    Args:
        x: projected features, [batch_size, D]
        temperature: scaling factor
    """
    device = x.device
    batch_size = x.shape[0]
    similarity_matrix = torch.matmul(x, x.T) / temperature

    # Create labels
    labels = torch.arange(batch_size).to(device)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    # Mask to remove similarity with itself
    mask = torch.eye(batch_size).to(device)
    labels = labels - mask

    # Compute loss
    exp_sim = torch.exp(similarity_matrix) * (1 - mask)
    log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True))
    loss = -(labels * log_prob).sum(1) / labels.sum(1)
    loss = loss.mean()
    return loss

# Pretraining Loop
def pretrain(model, dataloader, optimizer, epochs=10, device='cuda'):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            spectrogram = batch['spectrogram'].to(device)
            mfcc = batch['mfcc'].to(device)
            articulatory = batch['articulatory'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            projections = model(
                spectrogram, 
                mfcc, 
                articulatory, 
                input_ids, 
                attention_mask
            )
            loss = contrastive_loss(projections)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Pretraining Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# ------------------------------
# 8. Training and Evaluation
# ------------------------------

# Initialization of DataLoaders
def initialize_dataloaders():
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create dataset and dataloaders
    dataset = DysarthriaDataset(
        dataframe=df,
        tokenizer=tokenizer,
        transform=None  # Add torchvision transforms if needed
    )

    # Split into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4
    )

    return train_loader, val_loader, tokenizer

# Fine-Tuning Function
def fine_tune(model, dataloader, optimizer, criterion, scheduler, epochs=20, device='cuda'):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            spectrogram = batch['spectrogram'].to(device)
            mfcc = batch['mfcc'].to(device)
            articulatory = batch['articulatory'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                spectrogram, 
                mfcc, 
                articulatory, 
                input_ids, 
                attention_mask
            )
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, 
            all_preds, 
            average='binary'
        )
        print(
            f"Fine-Tuning Epoch [{epoch+1}/{epochs}], "
            f"Loss: {avg_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1: {f1:.4f}"
        )
        scheduler.step()

# Evaluation Function
def evaluate(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            spectrogram = batch['spectrogram'].to(device)
            mfcc = batch['mfcc'].to(device)
            articulatory = batch['articulatory'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                spectrogram, 
                mfcc, 
                articulatory, 
                input_ids, 
                attention_mask
            )
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_preds, 
        average='binary'
    )
    print(
        f"Evaluation - Accuracy: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, "
        f"F1: {f1:.4f}"
    )

# ------------------------------
# 9. Model Pruning and Quantization
# ------------------------------

def prune_model(model, amount=0.3):
    """
    Apply L1 unstructured pruning to Conv2d and Linear layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

def quantize_model(model):
    """
    Apply dynamic quantization to Linear and Conv2d layers.
    """
    model.eval()
    model_int8 = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear, nn.Conv2d}, 
        dtype=torch.qint8
    )
    return model_int8

# ------------------------------
# 10. Main Execution
# ------------------------------
def main():
    # Initialize dataloaders and tokenizer
    train_loader, val_loader, tokenizer = initialize_dataloaders()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize Encoder
    spectrogram_cnn = SpectrogramCNN()
    mfcc_rnn = MFCCRNN()
    text_transformer = TextTransformer()
    encoder = Encoder(spectrogram_cnn, mfcc_rnn, text_transformer)

    # Initialize SimCLR Pretrainer
    pretrainer = SimCLRPretrainer(encoder=encoder, projection_dim=128)
    pretrainer_optimizer = optim.Adam(pretrainer.parameters(), lr=1e-4)

    # Pretrain the model
    print("Starting Pretraining...")
    pretrain(pretrainer, train_loader, pretrainer_optimizer, epochs=20, device=device)

    # Initialize the classifier for fine-tuning
    classifier = DysarthriaClassifier(encoder=encoder, num_classes=2)

    # Optionally, transfer weights from pretrainer to classifier's encoder
    # Since both use the same encoder, weights are already shared

    # Define optimizer, criterion, and scheduler
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Fine-tune the model
    print("Starting Fine-Tuning...")
    fine_tune(
        classifier, 
        train_loader, 
        optimizer, 
        criterion, 
        scheduler, 
        epochs=30, 
        device=device
    )

    # Evaluate the model
    print("Evaluating on Validation Set...")
    evaluate(classifier, val_loader, device=device)

    # Prune the model
    print("Pruning the Model...")
    pruned_model = prune_model(classifier, amount=0.3)

    # Quantize the model
    print("Quantizing the Model...")
    quantized_model = quantize_model(pruned_model)

    # Save the models
    torch.save(classifier.state_dict(), 'dysarthria_classifier.pth')
    torch.save(quantized_model.state_dict(), 'dysarthria_classifier_quantized.pth')
    print("Models Saved Successfully.")

    # Optional: Export to ONNX
    # Uncomment the following block after ensuring you have a sample batch
    """
    sample_batch = next(iter(train_loader))
    export_to_onnx(
        quantized_model, 
        (
            sample_batch['spectrogram'].to(device),
            sample_batch['mfcc'].to(device),
            sample_batch['articulatory'].to(device),
            sample_batch['input_ids'].to(device),
            sample_batch['attention_mask'].to(device)
        ),
        filename='dysarthria_classifier.onnx'
    )
    """

if __name__ == "__main__":
    main()

# ------------------------------
# 11. Real-Time Deployment Considerations
# ------------------------------
# After training, you can further optimize the model for deployment using ONNX or TensorRT.
# Example of exporting to ONNX:


