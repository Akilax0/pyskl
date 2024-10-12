import torch
from torch import nn,optim
import matplotlib.pyplot as plt
from torchinfo import summary
import argparse

from data import prepare_data
# from data_loader import *
from vit import ViT
from utils import save_checkpoint, save_experiment

config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
    "batch_size": 32
}

# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0

class Trainer:
    
    def __init__(self,model,optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn 
        self.exp_name = exp_name
        self.device = device 
        
        
    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        # Training for number of epochs
        
        train_losses, test_losses, accuracies = [],[],[]
        
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            
            print(f"Epochs: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f},Accuracy: {accuracy:.4f}")
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 !=epochs:
                print("\tSave checkpoint at epoch",i+1)
                save_checkpoint(self.exp_name, self.model, i+1)
                
            save_experiment(self.exp_name,self.model, train_losses, test_losses, accuracies) 
                
            
    def train_epoch(self,trainloader):
        # Training for one epoch
        
        self.model.train()
        total_loss = 0
        # print(trainloader)
        for batch in trainloader: 
            # Move batch to device
            batch = [t.to(self.device) for t in batch]
            imgs, labels = batch
            # print(len(batch[0]))
            
            #zero grad 
            self.optimizer.zero_grad()
            
            # Calculate loss
            # print(self.model(imgs).shape, labels.shape)
            loss = self.loss_fn(self.model(imgs), labels)
            
            # Backprop
            loss.backward()
            
            # Update model param
            self.optimizer.step()
            total_loss += loss.item() * len(imgs)
            
        return total_loss / len(trainloader.dataset)
    

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                
                batch = [t.to(self.device) for t in batch]
                imgs, labels = batch
                
                logits = self.model(imgs)
                
                # Calculate loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(imgs)
                
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
                
        accuracy = correct / len(testloader.dataset)  
        avg_loss = total_loss / len(testloader.dataset)
        
        return accuracy, avg_loss
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name",type=str,  required=True)
    parser.add_argument("--batch-size",type=int, default=256)
    parser.add_argument("--epochs", type=float, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device",type=str)
    parser.add_argument("--save-model-every", type=int, default=0)
    
    args=parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    return args


def main():
    args = parse_args()
    
    # Trainina params
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device =args.device
    save_model_every_n_epochs = args.save_model_every
    
    # Load dataset
    # training_dataloader
    # test_dataloader    
   
    trainloader, testloader, _ = prepare_data(batch_size=batch_size) 
    # Create model , optim, loss func, trainer
    model = ViT(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model,optimizer, loss_fn, args.exp_name, device=device)
    trainer.train(trainloader=trainloader, testloader=testloader, epochs=epochs, save_model_every_n_epochs=save_model_every_n_epochs)

if __name__ == "__main__":
    main() 

# random_images, random_labels = next(iter(training_dataloader))

# patch_embedding_layer = PatchEmbeddingLayer(in_channels=IMAGE_CHANNELS, patch_size=PATCH_SIZE, \
#     embedding_dim=IMAGE_CHANNELS * PATCH_SIZE ** 2)

# patch_embeddings = patch_embedding_layer(random_images)
# print(patch_embeddings.shape)



# summary(model=patch_embedding_layer, input_size = (BATCH_SIZE, 3, 224, 224),
#         col_names = ["input_size", "output_size", "num_params", "trainable"],
#         col_width = 20,
#         row_settings=["var_names"])


# multihead_self_attention_block = MultiHeadSelfAttentionBlock(embedding_dims=EMBEDDING_DIMS,
#                                                              num_heads=  12)

# print(f'Shape of the input Patch Embeddings => {list(patch_embeddings.shape)} <== [batch_size, num_patches + 1, embedding_dims]')
# print(f'Shape of the output from MSA Block => {list(multihead_self_attention_block(patch_embeddings).shape)} <== [batch_size, num_patches + 1, embedding_dims]')

# summary(model=multihead_self_attention_block, input_size = (1, 197, 768), # (batch_size , num_patches, embedding_dimension)
#         col_names = ["input_size", "output_size", "num_params", "trainable"],
#         col_width = 20,
#         row_settings=["var_names"])

# mlp_block = MachineLearningPerceptronBlock(embedding_dims= EMBEDDING_DIMS,
#                                            mlp_size=3072,
#                                            mlp_dropout=0.1)

# summary(model=mlp_block, 
#         input_size = (1, 197, 768), # (batch_size , num_patches, embedding_dimension)
#         col_names = ["input_size", "output_size", "num_params", "trainable"],
#         col_width = 20,
#         row_settings=["var_names"])

# transformer_block = TransformerBlock(embedding_dims=EMBEDDING_DIMS,
#                                      mlp_dropout=0.1,
#                                      attn_dropout = 0.0,
#                                      mlp_size=3072,
#                                      num_heads = 12)

# print(f'Shape of the input Patch Embeddings => {list(patch_embeddings.shape)} <= [batch_size, num_patches+1, embedding_dims ]')
# print(f'Shape of the output from Transformer Block => {list(transformer_block(patch_embeddings).shape)} <= [batch_size, num_patches+1, embedding_dims ]')

# summary(model=transformer_block, 
#         input_size = (1, 197, 768), # (batch_size , num_patches, embedding_dimension)
#         col_names = ["input_size", "output_size", "num_params", "trainable"],
#         col_width = 20,
#         row_settings=["var_names"])


# vit_block = ViT(img_size=224,
#                 in_channels=3,
#                 patch_size=PATCH_SIZE,
#                 embedding_dims=EMBEDDING_DIMS,
#                 num_transformer_layers= 12,
#                 mlp_dropout=0.1,
#                 attn_dropout=0.0,
#                 mlp_size=3072,
#                 num_heads=12,
#                 num_classes=1000)

# summary(model=vit_block,
#         input_size=(BATCH_SIZE, 3, 224, 224), # (batch_size, num_patches, embedding_dimension)
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])