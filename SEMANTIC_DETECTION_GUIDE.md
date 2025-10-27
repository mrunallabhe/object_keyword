# Semantic Object Detection with CLIP

## Overview

This system extends YOLO detection with **CLIP (Contrastive Language–Image Pretraining)** to understand natural language queries with relationships and context.

## What CLIP Understands

CLIP can understand complex relationships and context, not just keyword matching:

### Examples

✅ **"person holding cup"** - Understands the action "holding"  
✅ **"dog sitting on sofa"** - Understands positional relationship  
✅ **"man wearing blue jeans"** - Understands appearance descriptions  
✅ **"woman with red bag"** - Understands possession  
✅ **"vehicle at intersection"** - Understands location context  

## How CLIP Works

### 1. Dual Encoding

CLIP uses two encoders:
- **Text Encoder**: Converts text query to embedding
- **Image Encoder**: Converts image to embedding

```
Text Query → CLIP Text Encoder → Text Embedding
Image → CLIP Image Encoder → Image Embedding
```

### 2. Similarity Calculation

CLIP calculates cosine similarity between embeddings:

```
similarity = cosine_similarity(image_embedding, text_embedding)
```

Higher similarity = better match for the query.

## Implementation

### Basic Usage

```python
from clip_detector import CLIPSemanticDetector

# Initialize
detector = CLIPSemanticDetector(model_name="ViT-B/32")

# Simple semantic search
is_match, similarity = detector.semantic_search("image.jpg", "person holding cup")
print(f"Match: {is_match}, Similarity: {similarity:.3f}")

# Detect objects with semantic queries
detections = detector.detect_semantic_objects(
    "image.jpg", 
    ["person holding cup", "dog on sofa"]
)
```

### In Web Interface

**Simple Query:**
```
keywords: "person, gun, vehicle"
```

**Semantic Query:**
```
keywords: "person holding weapon, dog sitting near person, vehicle at intersection"
```

## Supported Query Types

### 1. **Actions** (What is happening?)
- "person holding cup"
- "man carrying bag"
- "woman opening door"

### 2. **Positions** (Where is it?)
- "dog sitting on sofa"
- "person standing near car"
- "object on table"

### 3. **Appearance** (What does it look like?)
- "man wearing blue jeans"
- "person with red shirt"
- "woman with long hair"

### 4. **Possession** (Who has what?)
- "man with umbrella"
- "woman with red bag"
- "person holding phone"

### 5. **Location Context** (Where in the scene?)
- "vehicle at intersection"
- "person in doorway"
- "object on ground"

### 6. **Combinations** (Multiple context)
- "person wearing red shirt holding phone"
- "dog sitting on sofa near window"
- "vehicle at intersection with open door"

## Comparison: Simple vs Semantic

### Simple Keyword Matching (YOLO Only)
```
Query: "person"
Result: Detects any person in image
```

### Semantic Understanding (CLIP)
```
Query: "person holding cup"
Result: Only matches if person is actually holding a cup
```

## Usage Examples

### Example 1: Action Detection

```python
queries = ["person holding weapon", "person opening door", "person running"]
detections = detector.detect_semantic_objects(image_path, queries)
```

### Example 2: Appearance-Based Search

```python
queries = ["man wearing blue jeans", "person with red bag", "woman wearing hat"]
detections = detector.detect_semantic_objects(image_path, queries)
```

### Example 3: Spatial Relationships

```python
queries = ["dog sitting on sofa", "person standing near car", "object on table"]
detections = detector.detect_semantic_objects(image_path, queries)
```

## How It Works Technically

### 1. Text Embedding Generation

```python
# Tokenize and encode text
text_tokens = clip.tokenize(["person holding cup"])
text_features = model.encode_text(text_tokens)
```

### 2. Image Embedding Generation

```python
# Preprocess and encode image
image_tensor = preprocess(Image.open("image.jpg"))
image_features = model.encode_image(image_tensor)
```

### 3. Similarity Calculation

```python
# Calculate cosine similarity
similarity = cosine_similarity(image_features, text_features)
```

### 4. Object Localization (Sliding Window)

```python
# For precise localization:
# 1. Divide image into tiles
# 2. Calculate similarity for each tile
# 3. Keep tiles with high similarity
# 4. Apply NMS to remove overlaps
```

## Integration with YOLO

The system combines both approaches:

1. **YOLO** for quick, accurate object detection (80 classes)
2. **CLIP** for semantic understanding of relationships

### Hybrid Detection

```python
# Simple keywords → YOLO
keywords = ["person", "car", "gun"]

# Semantic queries → CLIP
semantic = ["person holding gun", "car at intersection"]

# Combined results
all_detections = yolo_results + clip_results
```

## Performance

### Speed
- **CLIP ViT-B/32**: ~2-5 FPS (slower but more accurate)
- **YOLO**: ~50 FPS (fast)
- **Combined**: ~2-5 FPS (due to CLIP bottleneck)

### Accuracy
- **Semantic understanding**: Very high for complex queries
- **Relationship detection**: Excellent for spatial/action queries
- **Context awareness**: Superior to keyword matching

## Query Tips

### ✅ Good Queries
- Be specific: "person holding red cup"
- Use relationships: "dog sitting on sofa"
- Describe actions: "man opening door"
- Include context: "vehicle at intersection"

### ❌ Avoid
- Too vague: "thing"
- Multiple unrelated objects: "person and cat and car" (separate queries)
- Negative queries: "person without hat" (CLIP struggles with negatives)

## Code Examples

### Example 1: Basic Semantic Search

```python
from clip_detector import CLIPSemanticDetector

detector = CLIPSemanticDetector()

# Search for semantic matches
is_match, score = detector.semantic_search("scene.jpg", "person holding cup")
if is_match:
    print(f"Match found! Similarity: {score:.3f}")
```

### Example 2: Multiple Queries

```python
queries = [
    "person holding weapon",
    "dog sitting on furniture",
    "vehicle at intersection"
]

detections = detector.detect_semantic_objects("image.jpg", queries)

for det in detections:
    print(f"{det['label']}: {det['score']:.2f} at {det['bbox']}")
```

### Example 3: Annotate Results

```python
detections = detector.detect_semantic_objects("input.jpg", ["person holding cup"])
detector.annotate_image("input.jpg", detections, "output.jpg")
```

## Web Interface Usage

### Simple Mode
Enter comma-separated keywords:
```
person, gun, vehicle
```

### Semantic Mode
Enter natural language queries (comma-separated):
```
person holding weapon, dog sitting on sofa, vehicle at intersection
```

The system automatically detects semantic queries (contains spaces or relationship words) and uses CLIP for those, YOLO for simple keywords.

## Advanced Features

### Custom Thresholds

```python
detector = CLIPSemanticDetector()

# Lower threshold = more results
detections = detector.detect_semantic_objects(
    "image.jpg",
    ["person holding cup"],
    threshold=0.15  # More sensitive
)
```

### Adjustable Tile Size

```python
# Smaller tiles = finer localization
detections = detector.detect_semantic_objects(
    "image.jpg",
    ["person"],
    tile_size=112,  # Smaller tiles
    stride=50       # More overlap
)
```

## Limitations

1. **Speed**: CLIP is slower than YOLO (2-5 FPS vs 50 FPS)
2. **Memory**: Requires more GPU memory
3. **Precision**: Sliding window approach is approximate, not pixel-perfect
4. **Negatives**: Can't handle "person without hat"

## Best Practices

1. **Combine both**: Use YOLO for speed, CLIP for semantic queries
2. **Be specific**: More specific queries = better results
3. **Test thresholds**: Adjust similarity threshold for your use case
4. **Batch queries**: Process multiple similar queries together

## Installation

CLIP is automatically installed with the system. If you need to install manually:

```bash
pip install torch torchvision
pip install ftfy regex tqdm
```

The CLIP library is included with `torch` and `torchvision`.

## Summary

- ✅ **Simple queries**: Use YOLO (fast)
- ✅ **Semantic queries**: Use CLIP (accurate)
- ✅ **Automatic detection**: System chooses the right method
- ✅ **Powerful relationships**: Understands context and actions

**The system now understands your natural language queries!** 🎉

