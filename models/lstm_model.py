import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple

class LSTMICD10Verifier(nn.Module):
    """
    LSTM model for verifying ICD-10 codes based on symptoms.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of LSTM hidden layer
            output_size: Size of output (typically 1 for binary classification)
            num_layers: Number of LSTM layers
        """
        super(LSTMICD10Verifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take output from the last time step
        out = self.fc(out[:, -1, :])
        
        # Apply sigmoid activation
        out = self.sigmoid(out)
        
        return out
    
    def predict(self, x, threshold=0.5):
        """
        Make binary predictions using the model.
        
        Args:
            x: Input tensor
            threshold: Threshold for binary classification
            
        Returns:
            Binary predictions
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            outputs = self(x)
            predictions = (outputs >= threshold).float()
            confidence = outputs
        
        return predictions, confidence
