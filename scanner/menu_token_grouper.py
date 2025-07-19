import tensorflow as tf
import numpy as np

class WineMenuGrouper:
    """
    A GNN-based wine menu grouper that follows sklearn-style API
    
    Usage:
        grouper = WineMenuGrouper()
        grouper.fit(training_datasets)
        wine_entries = grouper.predict(test_tokens)
    """
    
    def __init__(self, similarity_threshold=0.5, epochs=50, batch_size=1, validation_split=0.2):
        """
        Initialize the WineMenuGrouper
        
        Args:
            similarity_threshold: Threshold for grouping tokens (default: 0.5)
            epochs: Number of training epochs (default: 50)
            batch_size: Training batch size (default: 1)
            validation_split: Validation split ratio (default: 0.2)
        """
        self.similarity_threshold = similarity_threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model = None
        self.is_fitted = False
    
    def _prepare_features(self, tokens_list: list[dict]):
        """Extract features from tokens"""
        features = []
        for token in tokens_list:
            f = [
                token.get('x', 0) / 800.0,
                token.get('y', 0) / 1200.0,
                len(token.get('text', '')) / 20.0,
                1.0 if token.get('is_vintage') else 0.0,
                1.0 if token.get('is_price') else 0.0,
                1.0 if token.get('is_varietal_red') else 0.0,
                1.0 if token.get('is_varietal_white') else 0.0,
                1.0 if token.get('is_region') else 0.0,
            ]
            features.append(f)
        return np.array(features)

    def _create_adjacency(self, tokens_list):
        """Create adjacency matrix"""
        n = len(tokens_list)
        adj = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = tokens_list[i].get('x', 0) - tokens_list[j].get('x', 0)
                dy = tokens_list[i].get('y', 0) - tokens_list[j].get('y', 0)
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < 300:
                    strength = max(0, 1 - distance / 300)
                    adj[i, j] = strength
                    adj[j, i] = strength
        
        # Normalize
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        adj = adj / row_sums
        
        return adj

    def _flatten_dataset(self, wine_entries):
        """Convert grouped wine entries to (tokens, groups) format"""
        all_tokens = []
        groups = []
        
        for group_idx, group in enumerate(wine_entries):
            group_indices = []
            for token in group:
                token_idx = len(all_tokens)
                all_tokens.append(token)
                group_indices.append(token_idx)
            groups.append(group_indices)
        
        return all_tokens, groups

    def _tokens_to_entry(self, token_group):
        """Convert token group to wine entry"""
        entry = {'description': ''}
        sorted_tokens = sorted(token_group, key=lambda t: t.get('x', 0))
        
        for token in sorted_tokens:
            text = token.get('text', '').strip()
            
            if token.get('is_vintage'):
                try:
                    entry['year'] = int(text)
                except:
                    pass
            elif token.get('is_price'):
                try:
                    price = float(text.replace('$', '').replace(',', ''))
                    entry['price'] = max(entry.get('price', 0), price)
                except:
                    pass
            elif token.get('is_varietal_red'):
                entry['type'] = 'red'
                entry['varietal'] = text.lower()
            elif token.get('is_varietal_white'):
                entry['type'] = 'white'
                entry['varietal'] = text.lower()
            elif token.get('is_region'):
                entry['region'] = text.lower()
            elif token.get('is_country'):
                entry['country'] = text.lower()
            else:
                if entry['description']:
                    entry['description'] += ' ' + text
                else:
                    entry['description'] = text
        
        return entry

    def _build_model(self):
        """Build the GNN model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, 8)),
            tf.keras.layers.Dense(32, activation='relu'), 
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def fit(self, training_datasets, verbose=1):
        """
        Train the wine menu grouper
        
        Args:
            training_datasets: List of training datasets (each dataset is a list of wine entries)
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        
        Returns:
            self: Returns the instance for method chaining
        """
        print("🏋️ Training WineMenuGrouper...")
        
        # Build model
        self.model = self._build_model()
        
        # Prepare training data
        X_train = []
        y_train = []

        for dataset in training_datasets:
            tokens, groups = self._flatten_dataset(dataset)
            features = self._prepare_features(tokens)
            adjacency = self._create_adjacency(tokens)
            
            # Graph convolution
            x = np.dot(adjacency, features)
            
            # Create target similarity matrix
            n_tokens = len(tokens)
            target_similarity = np.zeros(n_tokens)
            
            # Mark tokens that belong to groups
            for group in groups:
                for i in group:
                    if i < n_tokens:
                        target_similarity[i] = 1.0
            
            X_train.append(x)
            y_train.append(target_similarity)

        # Pad to same length
        max_tokens = max(len(x) for x in X_train)
        X_padded = []
        y_padded = []

        for x, y in zip(X_train, y_train):
            # Pad X
            x_pad = np.zeros((max_tokens, x.shape[1]))
            x_pad[:len(x)] = x
            X_padded.append(x_pad)
            
            # Pad y
            y_pad = np.zeros(max_tokens)
            y_pad[:len(y)] = y
            y_padded.append(y_pad)

        X_train_array = np.array(X_padded)
        y_train_array = np.array(y_padded)

        print(f"📚 Training data shape: X={X_train_array.shape}, y={y_train_array.shape}")

        # Train the model
        history = self.model.fit(
            X_train_array, 
            y_train_array,
            epochs=self.epochs,
            validation_split=self.validation_split,
            batch_size=self.batch_size,
            verbose=verbose,
        )

        self.is_fitted = True
        print("✅ Training completed!")
        
        return self


    def predict(self, tokens):
        """
        Predict wine entries from tokens (high-level sklearn-style method)
        
        Args:
            tokens: List of token dictionaries with x, y, text, and feature flags
        
        Returns:
            List of wine entry dictionaries with parsed information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        
        # Prepare features and adjacency matrix
        features = self._prepare_features(tokens)
        adjacency = self._create_adjacency(tokens)
        
        # Graph convolution
        x = np.dot(adjacency, features)
        
        # Create padded input
        max_tokens = len(x)
        x_padded = np.zeros((max_tokens, x.shape[1]))
        x_padded[:len(x)] = x
        x_input = np.expand_dims(x_padded, 0)
        
        # Get model predictions (similarity scores per token)
        predictions = self.model.predict(x_input, verbose=0)
        similarity_scores = predictions[0][:len(tokens)]
        
        # Price-Anchored Grouping: Start from prices and collect related tokens
        groups = []
        
        # Find all price tokens first
        price_tokens = []
        for i, token in enumerate(tokens):
            if (token.get('is_price') and 
                similarity_scores[i] > self.similarity_threshold):
                price_tokens.append((i, token))
        
        # For each price token, collect its wine entry
        used_tokens = set()
        
        for price_idx, price_token in price_tokens:
            if price_idx in used_tokens:
                continue
                
            wine_entry = [price_token]
            used_tokens.add(price_idx)
            
            price_y = price_token.get('y', 0)
            price_x = price_token.get('x', 0)
            
            # Collect tokens that belong to this wine entry
            for i, token in enumerate(tokens):
                if i in used_tokens or i == price_idx:
                    continue
                    
                if similarity_scores[i] <= self.similarity_threshold:
                    continue
                    
                token_y = token.get('y', 0)
                token_x = token.get('x', 0)
                
                # Include tokens that are above the price and within reasonable bounds
                y_diff = price_y - token_y  # Positive if token is above price
                x_diff = abs(token_x - price_x)
                
                if (0 <= y_diff <= 100 and  # Token is above price, within 100px
                    x_diff <= 400):         # Not too far horizontally
                    wine_entry.append(token)
                    used_tokens.add(i)
            
            # Sort tokens in this wine entry by position
            wine_entry.sort(key=lambda t: (t.get('y', 0), t.get('x', 0)))
            
            if len(wine_entry) >= 2:  # Need at least price + one other token
                groups.append(wine_entry)

        # Convert to wine entries
        wine_entries = [self._tokens_to_entry(group) for group in groups]
        
        return wine_entries

    def set_params(self, **params):
        """Set parameters (sklearn-style)"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def get_params(self, deep=True):
        """Get parameters (sklearn-style)"""
        return {
            'similarity_threshold': self.similarity_threshold,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split
        }
