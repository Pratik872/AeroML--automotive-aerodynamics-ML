import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")



class TNet(nn.Module):
    """Transformation Network - learns spatial alignment"""

    def __init__(self, k=3):

        super(TNet, self).__init__()
        self.k = k

        #Shared MLPs
        self.conv1 = nn.Conv1d(k, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 256, 1)

        #Fully Connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, k*k)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

        # Initialize transformation as identity
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).view(-1))


    def forward(self, X):
        batch_size = X.size(0)

        #Shared MLPs

        X = F.relu(self.bn1(self.conv1(X)))
        X = F.relu(self.bn2(self.conv2(X)))
        X = F.relu(self.bn3(self.conv3(X)))

        #Max Pooling
        X = torch.max(X, 2, keepdim=True)[0]
        X = X.view(-1, 256)

        # Fully connected layers
        X = F.relu(self.bn4(self.fc1(X)))
        X = F.relu(self.bn5(self.fc2(X)))
        X = self.fc3(X)

        #Reshape to transformation Matrix
        iden = torch.eye(self.k).view(1, self.k*self.k).repeat(batch_size, 1)

        if X.is_cuda:
            iden = iden.cuda()

        X = X + iden
        X = X.view(-1, self.k, self.k)

        return X
    

class PointNetEncoder(nn.Module):
    """PointNet backbone for feature extraction"""

    def __init__(self, global_feat=True, feature_transform=True):
        super(PointNetEncoder, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        # Input Transformation(3D spatial)
        self.input_transform = TNet(k=3)

        #First set of MLPs
        self.conv1 = nn.Conv1d(6,32, 1)  #6D input(xyz + normals)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 256, 1)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(256)

        #Feature transformation(64D)
        if self.feature_transform:
            self.feature_transform_net = TNet(k=32)

    def forward(self, X):
        n_pts = X.size()[2]

        #Input transformation
        trans = self.input_transform(X[:,:3,:])  #Only xyz for transformation
        x_xyz = X[:, :3, :].transpose(2, 1)
        x_xyz = torch.bmm(x_xyz, trans)
        x_xyz = x_xyz.transpose(2, 1)

        #Combine transformed xyz with original normals
        X = torch.cat([x_xyz, X[:,3:,:]], dim=1)

        #First MLP
        X = F.relu(self.bn1(self.conv1(X)))

        #Feature transform
        if self.feature_transform:
            trans_feat = self.feature_transform_net(X)
            X = X.transpose(2,1)
            X = torch.bmm(X, trans_feat)
            X = X.transpose(2,1)

        else:
            trans_feat = None

        #Continue MLPs
        pointfeat = X
        X = F.relu(self.bn2(self.conv2(X)))
        X = self.bn3(self.conv3(X))

        #Max Pooling to get global feature
        X = torch.max(X, 2, keepdim=True)[0]
        X = X.view(-1, 256)

        if self.global_feat:
            return X, trans, trans_feat
        else:
            #For segmentation
            X = X.view(-1, 256, 1).repeat(1,1,n_pts)
            return torch.cat([X, pointfeat], 1), trans, trans_feat
        

class AeroDynamicPointNet(nn.Module):
    """PointNet for automotive drag coefficient prediction"""
    def __init__(self, feature_transform=True):
        super(AeroDynamicPointNet, self).__init__()
        self.feature_transform = feature_transform

        #PointNet Encoder
        self.encoder = PointNetEncoder(global_feat=True,
                                       feature_transform=feature_transform)
        
        #Regression Head
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)  #Single drag coefficient

        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)


    def forward(self, X):
        #Get globl features
        X, trans, trans_feat = self.encoder(X)

        #Regression layers
        X = F.relu(self.bn1(self.fc1(X)))
        X = self.dropout(X)
        X = F.relu(self.bn2(self.fc2(X)))
        X = self.dropout(X)
        X = self.fc3(X)

        return X, trans, trans_feat
    
class PointNetTrainer:
    """Training pipeline for PointNet"""

    def __init__(self, model, train_loader, val_loader, lr= 0.0001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.7)

        self.train_losses = []
        self.val_losses = []
        self.val_r2_scores = []

    def feature_transform_regularization(self, trans_feat):
        """Regularization for feature transformation matrix"""
        if trans_feat is None:
            return 0
        
        d = trans_feat.size()[1]
        I = torch.eye(d)[None, :, :]
        if trans_feat.is_cuda:
            I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans_feat, trans_feat.transpose(2,1)) - I, dim=(1,2)))
        return loss
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for points, targets in pbar:
            points = points.to(device).float()
            targets = targets.to(device).float().unsqueeze(1)

            self.optimizer.zero_grad()

            pred, trans, trans_feat = self.model(points)
            loss = self.criterion(pred, targets)

            #Add feature transformation regularization
            if trans_feat is not None:
                reg_loss = self.feature_transform_regularization(trans_feat)
                loss += 0.001*reg_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets_list = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for points, targets in pbar:
                points = points.to(device).float()
                targets = targets.to(device).float().unsqueeze(1)

                pred, _, _ = self.model(points)
                loss = self.criterion(pred, targets)

                total_loss += loss.item()
                predictions.extend(pred.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

                pbar.set_postfix({'Val Loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(self.val_loader)

        #Calculate Rsquare
        predictions = np.array(predictions).flatten()
        targets_list = np.array(targets_list).flatten()
        r2 = r2_score(targets_list, predictions)

        return avg_loss, r2, predictions, targets_list
    
    def train(self, epochs):
        print(f"Training PointNet on {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_r2 = -float('inf')
        best_model_state = None

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            #Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            #Validation
            val_loss, val_r2, preds, targets = self.validate()
            self.val_losses.append(val_loss)
            self.val_r2_scores.append(val_r2)

            #Scheduler step
            self.scheduler.step()

            #Save best model
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_model_state = self.model.state_dict().copy()

            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val R²: {val_r2:.2f}")

        #Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return best_r2
    
    def plot_training_curves(self):
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('Training Curves')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.val_r2_scores, 'g-')
        ax2.axhline(y=0.879, color='orange', linestyle='--', label='Baseline (87.9%)')
        ax2.axhline(y=0.90, color='red', linestyle='--', label='Target (90%)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Validation R²')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
