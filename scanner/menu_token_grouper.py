import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple

class MenuTokenGrouper:
    """
    Universal wine menu grouper that works on ANY layout:
    1. Groups spatially (primarily by Y, secondarily by X)
    2. Penalizes same token types (multiple vintages, etc)
    3. Automatically adapts to any menu layout without manual tuning
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def group_tokens(self, enriched_tokens: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """Group tokens spatially with semantic penalties"""
        
        if not enriched_tokens:
            return {}, []
        
        if self.debug:
            print(f"🍷 Grouping {len(enriched_tokens)} tokens...")
        
        # Step 1: Analyze layout to understand structure
        layout_info = self._analyze_layout(enriched_tokens)
        
        # Step 2: Prepare features based on layout
        features = self._prepare_universal_features(enriched_tokens, layout_info)
        
        # Step 3: Compute distances with semantic penalties
        distances = self._compute_distances(features, enriched_tokens)
        
        # Step 4: Find eps based on layout analysis
        eps = self._find_universal_eps(distances, enriched_tokens, layout_info)
        
        # Step 5: Cluster
        clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed').fit(distances)
        labels = clustering.labels_
        
        # Step 6: Group results
        groups = {}
        noise = []
        
        for i, label in enumerate(labels):
            if label == -1:
                noise.append(enriched_tokens[i])
            else:
                if label not in groups:
                    groups[label] = []
                groups[label].append(enriched_tokens[i])
        
        # Sort by X position within groups
        for group in groups.values():
            group.sort(key=lambda t: t['x'])
        
        if self.debug:
            print(f"📊 Found {len(groups)} groups, {len(noise)} noise tokens")
            print(f"📏 Used eps={eps:.2f}")
        
        return groups, noise
    
    def _analyze_layout(self, tokens: List[Dict]) -> Dict:
        """Analyze menu layout to understand its structure"""
        
        y_values = [t['y'] for t in tokens]
        x_values = [t['x'] for t in tokens]
        
        # Find natural row groups by clustering Y positions
        unique_y = sorted(set(y_values))
        
        # Group nearby Y positions (same row)
        rows = []
        current_row = [unique_y[0]]
        
        for i in range(1, len(unique_y)):
            gap = unique_y[i] - unique_y[i-1]
            # Adaptive threshold based on font sizes
            h_values = [t['h'] for t in tokens]
            avg_height = np.mean(h_values)
            threshold = max(3, avg_height * 0.7)  # Dynamic threshold
            
            if gap <= threshold:
                current_row.append(unique_y[i])
            else:
                rows.append(current_row)
                current_row = [unique_y[i]]
        rows.append(current_row)
        
        # Calculate typical row spacing
        if len(rows) > 1:
            row_centers = [np.mean(row) for row in rows]
            row_gaps = [row_centers[i+1] - row_centers[i] for i in range(len(row_centers)-1)]
            typical_row_spacing = np.median(row_gaps)
        else:
            typical_row_spacing = np.mean(h_values) * 2
        
        # Calculate typical column spacing
        unique_x = sorted(set(x_values))
        if len(unique_x) > 1:
            x_gaps = [unique_x[i+1] - unique_x[i] for i in range(len(unique_x)-1)]
            # Filter small gaps (same column, slight misalignment)
            significant_gaps = [gap for gap in x_gaps if gap > 15]
            typical_col_spacing = np.median(significant_gaps) if significant_gaps else 100
        else:
            typical_col_spacing = 100
        
        layout_info = {
            'num_rows': len(rows),
            'num_tokens': len(tokens),
            'tokens_per_row': len(tokens) / len(rows),
            'typical_row_spacing': typical_row_spacing,
            'typical_col_spacing': typical_col_spacing,
            'y_range': max(y_values) - min(y_values),
            'x_range': max(x_values) - min(x_values),
            'avg_font_size': np.mean(h_values)
        }
        
        if self.debug:
            print(f"📐 Layout: {layout_info['num_rows']} rows, {layout_info['tokens_per_row']:.1f} tokens/row")
            print(f"📐 Spacing: rows={layout_info['typical_row_spacing']:.1f}px, cols={layout_info['typical_col_spacing']:.1f}px")
        
        return layout_info
    
    def _prepare_universal_features(self, tokens: List[Dict], layout_info: Dict) -> np.ndarray:
        """Prepare features that work universally across layouts"""
        
        # UNIVERSAL SCALING PRINCIPLE:
        # - Scale so typical row spacing becomes a target value (e.g., 10 units)
        # - This ensures different layouts have similar feature spaces
        
        target_row_spacing = 10.0  # Target units between rows
        target_col_span = 5.0      # Target units across columns
        
        row_spacing = layout_info['typical_row_spacing']
        col_spacing = layout_info['x_range']
        
        # Scale based on actual layout measurements
        y_scale = max(1.0, row_spacing / target_row_spacing)
        x_scale = max(1.0, col_spacing / target_col_span)
        
        features = []
        for token in tokens:
            # Normalized coordinates
            y_norm = token['y'] / y_scale
            x_norm = token['x'] / x_scale * 0.3  # X secondary to Y
            features.append([y_norm, x_norm])
        
        if self.debug:
            y_span = max(f[0] for f in features) - min(f[0] for f in features)
            x_span = max(f[1] for f in features) - min(f[1] for f in features)
            actual_row_spacing = row_spacing / y_scale
            print(f"📏 Normalized: row_spacing={actual_row_spacing:.1f}, Y_span={y_span:.1f}, X_span={x_span:.1f}")
        
        return np.array(features)
    
    def _compute_distances(self, features: np.ndarray, tokens: List[Dict]) -> np.ndarray:
        """Compute pairwise distances with semantic penalties"""
        
        n = len(features)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Spatial distance
                spatial_dist = np.linalg.norm(features[i] - features[j])
                
                # Semantic penalty (much larger than spatial distances)
                penalty = self._semantic_penalty(tokens[i], tokens[j])
                
                total_dist = spatial_dist + penalty
                distances[i, j] = total_dist
                distances[j, i] = total_dist
        
        return distances
    
    def _semantic_penalty(self, token1: Dict, token2: Dict) -> float:
        """Calculate penalty for grouping two tokens of same semantic type"""
        
        penalty = 0
        
        # Large penalties (much larger than spatial distances ~1-20)
        if token1.get('is_vintage') and token2.get('is_vintage'):
            penalty += 1000
        
        if token1.get('is_varietal_red') and token2.get('is_varietal_red'):
            penalty += 1000
            
        if token1.get('is_varietal_white') and token2.get('is_varietal_white'):
            penalty += 1000
        
        if token1.get('is_varietal_other') and token2.get('is_varietal_other'):
            penalty += 1000
        
        # Mixed varietals
        red_white_mix = ((token1.get('is_varietal_red') and token2.get('is_varietal_white')) or 
                        (token1.get('is_varietal_white') and token2.get('is_varietal_red')))
        if red_white_mix:
            penalty += 1000
        
        # Medium penalty for multiple prices (glass/bottle prices allowed)
        if token1.get('is_price') and token2.get('is_price'):
            penalty += 200
        
        # Headers
        headers = ['WHITE WINES', 'RED WINES', 'WINE LIST', 'YEAR', 'WINE', 'REGION', 'GLASS', 'BOTTLE', 'PRICE']
        if token1['text'].upper() in headers and token2['text'].upper() in headers:
            penalty += 2000
        
        return penalty
    
    def _find_universal_eps(self, distances: np.ndarray, tokens: List[Dict], layout_info: Dict) -> float:
        """Find eps that works universally by analyzing distance distribution"""
        
        # Get spatial-only distances (no penalties)
        spatial_distances = []
        
        features = self._prepare_universal_features(tokens, layout_info)
        
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                penalty = self._semantic_penalty(tokens[i], tokens[j])
                if penalty == 0:  # Only spatial distance
                    spatial_dist = np.linalg.norm(features[i] - features[j])
                    spatial_distances.append(spatial_dist)
        
        if not spatial_distances:
            return 5.0  # Fallback
        
        spatial_distances = np.array(spatial_distances)
        
        if self.debug:
            print(f"📊 Spatial distances: {len(spatial_distances)} pairs")
            print(f"📊 Range: {np.min(spatial_distances):.2f} to {np.max(spatial_distances):.2f}")
            print(f"📊 Percentiles: 25th={np.percentile(spatial_distances, 25):.2f}, 50th={np.percentile(spatial_distances, 50):.2f}")
        
        # UNIVERSAL EPS STRATEGY:
        # Use a percentile that captures same-row tokens but excludes cross-row tokens
        # This should work regardless of menu size or layout
        
        # For normalized features where row spacing ≈ 10 units:
        # - Same row tokens: 0-5 units apart
        # - Different row tokens: 10+ units apart
        # - Target eps: ~6-8 units (captures same row, excludes different rows)
        
        target_eps = min(8.0, np.percentile(spatial_distances, 40))
        
        if self.debug:
            print(f"📊 Target eps: {target_eps:.2f}")
        
        return target_eps

