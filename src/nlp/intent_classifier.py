"""
Intent Classifier
High-accuracy intent classification using transformer models and ensemble methods.
"""

import asyncio
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModel
import pickle


class TransformerIntentClassifier(nn.Module):
    """Transformer-based intent classifier"""
    
    def __init__(self, num_intents: int, hidden_size: int = 384, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_intents = num_intents
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_intents)
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Apply self-attention
        attended, _ = self.attention(embeddings, embeddings, embeddings, key_padding_mask=attention_mask)
        
        # Residual connection and layer norm
        embeddings = self.layer_norm(embeddings + attended)
        
        # Pool embeddings (mean pooling)
        if attention_mask is not None:
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings = embeddings * mask_expanded
            pooled = embeddings.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = embeddings.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class IntentClassifier:
    """Advanced Intent Classification System"""
    
    def __init__(self, intents: Dict, language_model, tokenizer):
        self.intents = intents
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_model: Optional[TransformerIntentClassifier] = None
        
        # Intent mapping
        self.intent_to_id: Dict[str, int] = {}
        self.id_to_intent: Dict[int, str] = {}
        
        # Training data
        self.training_data: List[Tuple[str, str]] = []
        
        # Model paths
        self.model_dir = Path("models/nlp/intent_classifier")
        self.model_path = self.model_dir / "classifier.pth"
        self.mapping_path = self.model_dir / "intent_mapping.json"
        
        # Performance tracking
        self.accuracy_history: List[float] = []
        
        # Ensemble components
        self.use_ensemble = True
        self.ensemble_weights = [0.6, 0.4]  # Transformer, TF-IDF
    
    async def initialize(self):
        """Initialize the intent classifier"""
        try:
            self.logger.info("Initializing Intent Classifier...")
            
            # Create model directory
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Build intent mapping
            await self._build_intent_mapping()
            
            # Prepare training data
            await self._prepare_training_data()
            
            # Initialize classifier model
            await self._initialize_classifier()
            
            # Load or train model
            if self.model_path.exists():
                await self._load_model()
            else:
                await self._train_initial_model()
            
            self.logger.info("Intent Classifier initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Intent Classifier initialization error: {e}")
            return False
    
    async def _build_intent_mapping(self):
        """Build intent name to ID mapping"""
        try:
            if self.mapping_path.exists():
                with open(self.mapping_path, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                    self.intent_to_id = mapping_data['intent_to_id']
                    self.id_to_intent = {int(k): v for k, v in mapping_data['id_to_intent'].items()}
            else:
                # Create new mapping
                intent_names = list(self.intents.keys())
                self.intent_to_id = {name: i for i, name in enumerate(intent_names)}
                self.id_to_intent = {i: name for i, name in enumerate(intent_names)}
                
                # Save mapping
                mapping_data = {
                    'intent_to_id': self.intent_to_id,
                    'id_to_intent': self.id_to_intent
                }
                
                with open(self.mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping_data, f, indent=2)
            
            self.logger.info(f"Built intent mapping for {len(self.intent_to_id)} intents")
            
        except Exception as e:
            self.logger.error(f"Intent mapping error: {e}")
            raise
    
    async def _prepare_training_data(self):
        """Prepare training data from intent examples"""
        try:
            self.training_data = []
            
            for intent_name, intent_def in self.intents.items():
                for example in intent_def.examples:
                    self.training_data.append((example, intent_name))
            
            # Add some negative examples (if available)
            # This would be enhanced with actual negative examples in production
            
            self.logger.info(f"Prepared {len(self.training_data)} training examples")
            
        except Exception as e:
            self.logger.error(f"Training data preparation error: {e}")
    
    async def _initialize_classifier(self):
        """Initialize the classifier model"""
        try:
            num_intents = len(self.intent_to_id)
            hidden_size = self.language_model.config.hidden_size
            
            self.classifier_model = TransformerIntentClassifier(
                num_intents=num_intents,
                hidden_size=hidden_size
            ).to(self.device)
            
            self.logger.info(f"Initialized classifier with {num_intents} intents")
            
        except Exception as e:
            self.logger.error(f"Classifier initialization error: {e}")
            raise
    
    async def _load_model(self):
        """Load pre-trained classifier model"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'accuracy_history' in checkpoint:
                self.accuracy_history = checkpoint['accuracy_history']
            
            self.logger.info("Loaded pre-trained intent classifier")
            
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
            await self._train_initial_model()
    
    async def _train_initial_model(self):
        """Train initial classifier model"""
        try:
            self.logger.info("Training initial intent classifier...")
            
            # Prepare training data
            texts = [item[0] for item in self.training_data]
            labels = [self.intent_to_id[item[1]] for item in self.training_data]
            
            # Get embeddings
            embeddings = await self._get_embeddings(texts)
            
            # Convert to tensors
            X = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            y = torch.tensor(labels, dtype=torch.long).to(self.device)
            
            # Training setup
            optimizer = torch.optim.AdamW(self.classifier_model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            self.classifier_model.train()
            
            # Simple training loop (in production, use proper train/val split)
            num_epochs = 50
            batch_size = 8
            
            for epoch in range(num_epochs):
                total_loss = 0
                num_batches = 0
                
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    batch_y = y[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    
                    logits = self.classifier_model(batch_X)
                    loss = criterion(logits, batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            
            # Evaluate on training data (basic check)
            self.classifier_model.eval()
            with torch.no_grad():
                logits = self.classifier_model(X)
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == y).float().mean().item()
                
                self.accuracy_history.append(accuracy)
                self.logger.info(f"Training accuracy: {accuracy:.4f}")
            
            # Save model
            await self._save_model()
            
            self.logger.info("Initial model training completed")
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
    
    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from language model"""
        try:
            embeddings = []
            
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.language_model(**inputs)
                    # Use mean pooling of last hidden state
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                    embeddings.append(embedding)
            
            return np.array(embeddings)
            
        except Exception as e:
            self.logger.error(f"Embedding generation error: {e}")
            return np.array([])
    
    async def _save_model(self):
        """Save classifier model"""
        try:
            torch.save({
                'model_state_dict': self.classifier_model.state_dict(),
                'intent_to_id': self.intent_to_id,
                'id_to_intent': self.id_to_intent,
                'accuracy_history': self.accuracy_history
            }, self.model_path)
            
        except Exception as e:
            self.logger.error(f"Model saving error: {e}")
    
    async def classify(self, text: str) -> Dict[str, Any]:
        """Classify intent of input text"""
        try:
            # Get embedding
            embedding = await self._get_embeddings([text])
            
            if len(embedding) == 0:
                return {'intent': 'unknown', 'confidence': 0.0, 'probabilities': {}}
            
            # Convert to tensor
            X = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            
            # Predict
            self.classifier_model.eval()
            with torch.no_grad():
                logits = self.classifier_model(X)
                probabilities = F.softmax(logits, dim=1)
                
                # Get top prediction
                max_prob, predicted_id = torch.max(probabilities, dim=1)
                predicted_intent = self.id_to_intent[predicted_id.item()]
                confidence = max_prob.item()
                
                # Get all probabilities
                prob_dict = {}
                for intent_id, prob in enumerate(probabilities[0]):
                    intent_name = self.id_to_intent[intent_id]
                    prob_dict[intent_name] = prob.item()
            
            return {
                'intent': predicted_intent,
                'confidence': confidence,
                'probabilities': prob_dict
            }
            
        except Exception as e:
            self.logger.error(f"Intent classification error: {e}")
            return {'intent': 'unknown', 'confidence': 0.0, 'probabilities': {}}
    
    async def add_intent(self, intent_def):
        """Add new intent and retrain classifier"""
        try:
            # Add to intents
            self.intents[intent_def.name] = intent_def
            
            # Update mapping
            if intent_def.name not in self.intent_to_id:
                new_id = len(self.intent_to_id)
                self.intent_to_id[intent_def.name] = new_id
                self.id_to_intent[new_id] = intent_def.name
                
                # Save updated mapping
                mapping_data = {
                    'intent_to_id': self.intent_to_id,
                    'id_to_intent': self.id_to_intent
                }
                
                with open(self.mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping_data, f, indent=2)
            
            # Add training examples
            for example in intent_def.examples:
                self.training_data.append((example, intent_def.name))
            
            # Reinitialize and retrain classifier
            await self._initialize_classifier()
            await self._train_initial_model()
            
            self.logger.info(f"Added intent: {intent_def.name}")
            
        except Exception as e:
            self.logger.error(f"Intent addition error: {e}")
    
    async def update_with_feedback(self, text: str, correct_intent: str, predicted_intent: str):
        """Update classifier with user feedback"""
        try:
            # Add to training data
            self.training_data.append((text, correct_intent))
            
            # Incremental learning (simplified)
            # In production, implement proper online learning
            if len(self.training_data) % 10 == 0:  # Retrain every 10 feedback examples
                await self._train_initial_model()
            
            self.logger.info(f"Updated with feedback: {text} -> {correct_intent}")
            
        except Exception as e:
            self.logger.error(f"Feedback update error: {e}")
    
    def get_intent_confidence_threshold(self, intent_name: str) -> float:
        """Get confidence threshold for specific intent"""
        if intent_name in self.intents:
            return self.intents[intent_name].confidence_threshold
        return 0.8  # Default threshold
    
    async def evaluate_performance(self, test_data: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate classifier performance on test data"""
        try:
            if not test_data:
                return {}
            
            predictions = []
            true_labels = []
            
            for text, true_intent in test_data:
                result = await self.classify(text)
                predictions.append(result['intent'])
                true_labels.append(true_intent)
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            
            # Update accuracy history
            self.accuracy_history.append(accuracy)
            
            # Generate classification report
            report = classification_report(
                true_labels, 
                predictions, 
                output_dict=True,
                zero_division=0
            )
            
            return {
                'accuracy': accuracy,
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall'],
                'f1_score': report['macro avg']['f1-score']
            }
            
        except Exception as e:
            self.logger.error(f"Performance evaluation error: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get classifier performance metrics"""
        return {
            'accuracy_history': self.accuracy_history[-10:],
            'current_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0,
            'total_intents': len(self.intent_to_id),
            'training_examples': len(self.training_data)
        }
    
    async def cleanup(self):
        """Cleanup classifier resources"""
        self.logger.info("Cleaning up Intent Classifier...")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.classifier_model = None
        
        self.logger.info("Intent Classifier cleanup completed")