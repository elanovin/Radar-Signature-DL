import torch
import numpy as np
from src.models.model import RadarClassifier
from src.data.preprocessing import process_radar_signal
from typing import Dict, Union

class RadarInference:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.classes = ['drone', 'bird', 'vehicle', 'human']
    
    def _load_model(self, model_path: str) -> RadarClassifier:
        """Load the trained model."""
        model = RadarClassifier(input_channels=1, num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, radar_signal: np.ndarray) -> Dict[str, Union[str, float]]:
        """
        Predict the class of a radar signal.
        
        Args:
            radar_signal: Raw radar signal data
            
        Returns:
            Dictionary containing predicted class and confidence
        """
        # Preprocess the signal
        spectrogram = process_radar_signal(radar_signal)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'class': self.classes[predicted_class],
            'confidence': confidence
        }

if __name__ == '__main__':
    # Example usage
    model = RadarInference('models/final_model.pth')
    
    # Example radar signal (replace with actual data)
    dummy_signal = np.random.randn(1000)
    
    prediction = model.predict(dummy_signal)
    print(f"Predicted class: {prediction['class']}")
    print(f"Confidence: {prediction['confidence']:.2f}") 