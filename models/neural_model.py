# models/neural_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
import os
import asyncio
from tqdm import tqdm

class TradingNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(TradingNeuralNetwork, self).__init__()
        
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.softmax(self.output(x))
        return x

class NeuralModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('models.neural')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler()
        self.model = None
        self.input_size = None
        
    def _calculate_features(self, df):
        """Calculate technical features with modern fillna usage"""
        features = pd.DataFrame(index=df.index)
        
        # Price based features
        features['returns'] = df['close'].pct_change()
        features['returns'] = features['returns'].fillna(0)
        
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['log_returns'] = features['log_returns'].fillna(0)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            ma = df['close'].rolling(window=window).mean()
            features[f'ma_{window}'] = ma.bfill()
            ma_ratio = df['close'] / ma
            features[f'ma_ratio_{window}'] = ma_ratio.fillna(1)
        
        # Volatility
        vol = features['returns'].rolling(window=20).std()
        features['volatility'] = vol.fillna(0)
        
        # Volume features
        vol_ma = df['tick_volume'].rolling(window=20).mean()
        features['volume_ma'] = vol_ma.bfill()
        vol_ratio = df['tick_volume'] / vol_ma
        features['volume_ratio'] = vol_ratio.fillna(1)
        
        return features

    async def prepare_data(self, df):
        """Prepare data for neural network"""
        try:
            # Calculate features
            features = self._calculate_features(df)
            if features.empty:
                raise ValueError("Failed to calculate features")
            
            # Create targets
            future_returns = df['close'].pct_change().shift(-1).fillna(0)
            targets = pd.Series(0, index=df.index)
            targets[future_returns > 0.001] = 1  # Buy signal
            targets[future_returns < -0.001] = 2  # Sell signal
            
            # Remove any remaining NaN values
            features = features.fillna(0)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            return torch.FloatTensor(scaled_features), torch.LongTensor(targets.values)
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            return None, None

    async def train(self, df, epochs=100, batch_size=32):
        """Train the neural network with progress tracking"""
        try:
            self.logger.info("1/4: Preparing data...")
            X, y = await self.prepare_data(df)
            if X is None or y is None:
                raise ValueError("Failed to prepare training data")
            
            self.logger.info("2/4: Initializing model...")
            if self.model is None:
                self.input_size = X.shape[1]
                self.model = TradingNeuralNetwork(self.input_size).to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            self.logger.info("3/4: Training model...")
            self.model.train()
            n_batches = len(X) // batch_size
            
            progress_bar = tqdm(range(epochs), desc="Training Neural Network")
            for epoch in progress_bar:
                total_loss = 0
                
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size
                    
                    batch_X = X[start_idx:end_idx].to(self.device)
                    batch_y = y[start_idx:end_idx].to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    if i % 10 == 0:
                        await asyncio.sleep(0)
                
                avg_loss = total_loss / n_batches
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            self.logger.info("4/4: Saving model...")
            await self.save_model()
            
            # Calculate and log final metrics
            self.model.eval()
            with torch.no_grad():
                final_outputs = self.model(X.to(self.device))
                final_loss = criterion(final_outputs, y.to(self.device))
                predictions = final_outputs.argmax(dim=1).cpu()
                accuracy = (predictions == y).float().mean().item()
                
            self.logger.info("\nTraining Results:")
            self.logger.info(f"- Final Loss: {final_loss.item():.4f}")
            self.logger.info(f"- Accuracy: {accuracy:.2%}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return False

    async def predict(self, df):
        """Make predictions using the trained model"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
                
            X, _ = await self.prepare_data(df)
            if X is None:
                raise ValueError("Failed to prepare prediction data")
                
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                X = X.to(self.device)
                outputs = self.model(X)
                predictions = outputs.cpu().numpy()
                
            # Convert to trading signals
            signals = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
            
            return {
                'signal': signals,
                'confidence': confidence,
                'probabilities': predictions
            }
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return None
            
    async def save_model(self):
        """Save the trained model"""
        try:
            if self.model is not None:
                os.makedirs('data/models', exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'input_size': self.input_size,
                    'scaler': self.scaler
                }, 'data/models/neural_model.pth')
                self.logger.info("Neural model saved successfully")
                return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
            
    async def load_model(self):
        """Load a trained model"""
        try:
            model_path = 'data/models/neural_model.pth'
            if not os.path.exists(model_path):
                self.logger.info("No saved neural model found")
                return False
                
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.input_size = checkpoint['input_size']
            self.model = TradingNeuralNetwork(self.input_size).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            
            self.logger.info("Neural model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False