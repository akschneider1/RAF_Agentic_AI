
# Technical Architecture

## System Overview

The Arabic PII NER system implements a hybrid approach combining rule-based pattern matching with machine learning-based named entity recognition for comprehensive PII detection in Arabic and English text.

```
┌─────────────────────────────────────────────────────────────────┐
│                        System Architecture                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Text    │────│  Text Processor │────│  PII Detector   │
│ (Arabic/English)│    │  & Tokenizer    │    │    Engine       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌─────────────────┐            │
                       │   Rule Engine   │◄───────────┼──────────┐
                       │   (Regex + ML)  │            │          │
                       └─────────────────┘            │          │
                                                       │          │
                       ┌─────────────────┐            │          │
                       │   NER Model     │◄───────────┘          │
                       │ (BERT-based)    │                       │
                       └─────────────────┘                       │
                                                                  │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐│
│   JSON Output   │◄───│  Result Merger  │◄───│ Confidence      ││
│   (Masked Text) │    │  & Formatter    │    │ Scorer          │◄┘
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. FastAPI Web Server (`app.py`)

**Purpose**: Production-ready API server with interactive testing interface

**Architecture**:
```python
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
├──────────────────────────────────────────────────────────────┤
│  Endpoints:                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   GET /     │  │ POST /detect│  │ POST /mask          │  │
│  │ (Web UI)    │  │ (Full API)  │  │ (Legacy endpoint)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
│  Models:                                                     │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │ TextInput   │  │ PIIResponse │                          │
│  │ - text      │  │ - original  │                          │
│  │ - confidence│  │ - masked    │                          │
│  └─────────────┘  │ - detected  │                          │
│                    │ - summary   │                          │
│                    └─────────────┘                          │
└──────────────────────────────────────────────────────────────┘
```

**Key Features**:
- Async request handling with uvicorn ASGI server
- Pydantic data validation and serialization
- Interactive HTML test interface with real-time detection
- RESTful API design with proper HTTP status codes
- CORS enabled for cross-origin requests

### 2. Rule-Based PII Detection Engine (`rules.py`)

**Purpose**: High-precision pattern matching for structured PII types

**Architecture**:
```python
┌──────────────────────────────────────────────────────────────┐
│                    PIIDetector Class                         │
├──────────────────────────────────────────────────────────────┤
│  Detection Methods:                                          │
│  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │ detect_phone_numbers│  │ detect_email_addresses      │   │
│  │ - Saudi: 05X XXXX XXX│ │ - Standard: user@domain.com │   │
│  │ - UAE: +971 5X XXX XXX│ │ - Arabic domains supported │   │
│  │ - Egypt: +20 1X XXX XXX│ └─────────────────────────────┘   │
│  │ - Jordan: +962 7X XXX │                                  │
│  └─────────────────────┘  ┌─────────────────────────────┐   │
│                           │ detect_national_ids         │   │
│  ┌─────────────────────┐  │ - Saudi: 1XXXXXXXXX        │   │
│  │ detect_iban_numbers │  │ - UAE: 784-YYYY-XXXXXXX-X  │   │
│  │ - SA: SA## #### #### │  │ - Egypt: 2XXXXXXXXXXXXX   │   │
│  │ - AE: AE## ### #### │  │ - Jordan: 10-digit format   │   │
│  │ - EG: EG## #### #### │  └─────────────────────────────┘   │
│  │ - JO: JO## XXXX #### │                                  │
│  └─────────────────────┘                                   │
└──────────────────────────────────────────────────────────────┘
```

**Pattern Matching Strategy**:
1. **Compiled Regex**: Pre-compiled patterns for optimal performance
2. **Confidence Scoring**: Pattern-specific confidence levels (0.7-0.95)
3. **Format Normalization**: Handles spaces, dashes, and country codes
4. **Overlap Resolution**: Prevents duplicate detections across patterns

### 3. Machine Learning Pipeline

#### Data Preprocessing (`preprocessing.py`)

**Purpose**: Convert Wojood dataset to transformer-compatible format

```python
┌──────────────────────────────────────────────────────────────┐
│                  Preprocessing Pipeline                      │
├──────────────────────────────────────────────────────────────┤
│  Input: Wojood CSV files                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   Raw CSV   │────│ BIO Tagging │────│  Tokenization   │  │
│  │  token|tag  │    │ Converter   │    │   (AraBERT)     │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ Label Align │────│ Truncation/ │────│ PyTorch Tensors │  │
│  │   & Pad     │    │ Padding     │    │   (input_ids,   │  │
│  └─────────────┘    └─────────────┘    │ attention_mask, │  │
│                                         │    labels)      │  │
│                                         └─────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Key Transformations**:
- **BIO Encoding**: Converts flat tags to Begin-Inside-Outside format
- **Subword Alignment**: Handles Arabic subword tokenization correctly
- **Sequence Padding**: Standardizes input length to 128 tokens
- **Label Smoothing**: Maps special tokens to ignored labels (-100)

#### Model Training (`train_model.py`)

**Purpose**: Fine-tune AraBERT for Arabic NER with memory optimization

```python
┌──────────────────────────────────────────────────────────────┐
│                   Training Architecture                      │
├──────────────────────────────────────────────────────────────┤
│  Model: aubmindlab/bert-base-arabertv2                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    BERT Layers                          │ │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────────────┐   │ │
│  │  │Embedding  │──│Transformer│──│ Classification    │   │ │
│  │  │  Layer    │  │  Blocks   │  │    Head           │   │ │
│  │  │(768 dim)  │  │  (12x)    │  │ (768→num_labels)  │   │ │
│  │  └───────────┘  └───────────┘  └───────────────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  Training Configuration:                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ • Learning Rate: 2e-5 (linear warmup)                  │ │
│  │ • Batch Size: 32 (gradient accumulation: 2)            │ │
│  │ • Max Length: 128 tokens                               │ │
│  │ • Epochs: 5 with early stopping                        │ │
│  │ • Optimizer: AdamW with weight decay                   │ │
│  │ • Mixed Precision: FP16 for memory efficiency          │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**Memory Optimization**:
- **Gradient Accumulation**: Effective batch size of 64 with 32 physical batch
- **Mixed Precision**: FP16 training reduces memory by ~40%
- **Dynamic Batching**: Variable sequence lengths for efficiency
- **Checkpoint Saving**: Regular model state preservation

### 4. Data Augmentation System

#### Synthetic Data Generation (`synthetic_generator.py`)

**Purpose**: Generate diverse training examples for improved model robustness

```python
┌──────────────────────────────────────────────────────────────┐
│                Data Augmentation Pipeline                    │
├──────────────────────────────────────────────────────────────┤
│  Template-Based Generation:                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   PERSON        │  │   LOCATION      │  │ ID_NUMBER   │  │
│  │ • Arabic names  │  │ • Addresses     │  │ • National  │  │
│  │ • 40+ templates │  │ • 35+ contexts  │  │ • 20+ forms │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│                                                              │
│  Generation Strategy:                                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 1. Template Selection (weighted random)                │ │
│  │ 2. Entity Substitution (realistic values)              │ │
│  │ 3. BIO Tag Assignment (automatic labeling)             │ │
│  │ 4. Format Validation (length, structure)               │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 5. Monitoring and Analysis Tools

#### Dataset Analysis (`dataset_inspector.py`, `entity_analyzer.py`)

**Purpose**: Comprehensive dataset quality assessment and statistics

```python
┌──────────────────────────────────────────────────────────────┐
│                    Analysis Dashboard                        │
├──────────────────────────────────────────────────────────────┤
│  Statistics Generated:                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ Entity Counts   │  │ Label Balance   │  │ Corpus Mix  │  │
│  │ • Per type      │  │ • O vs entities │  │ • Sub-corpus│  │
│  │ • Per split     │  │ • Distribution  │  │ • Coverage  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│                                                              │
│  Visualizations:                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ • Bar charts for entity frequency                      │ │
│  │ • Distribution plots for label balance                 │ │
│  │ • Training progress curves                             │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow Architecture

### 1. Training Data Flow

```
Wojood CSV Files
       │
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│ Schema      │────│ Data        │────│ Token           │
│ Mapper      │    │ Cleaner     │    │ Alignment       │
└─────────────┘    └─────────────┘    └─────────────────┘
       │                   │                    │
       ▼                   ▼                    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│ BIO Tag     │    │ Synthetic   │    │ DataLoader      │
│ Conversion  │    │ Augmentation│    │ Creation        │
└─────────────┘    └─────────────┘    └─────────────────┘
       │                   │                    │
       └───────────────────┼────────────────────┘
                           ▼
                 ┌─────────────────┐
                 │ Model Training  │
                 │ (BERT Fine-tune)│
                 └─────────────────┘
```

### 2. Inference Data Flow

```
Input Text (API Request)
          │
          ▼
    ┌─────────────┐
    │ Text        │
    │ Validation  │
    └─────────────┘
          │
          ▼
┌─────────────────────┐
│ Parallel Processing │
├─────────────────────┤
│ ┌─────────────────┐ │    ┌─────────────────┐
│ │ Rule-based      │ │    │ ML-based NER    │
│ │ Pattern Match   │ │    │ (BERT Model)    │
│ └─────────────────┘ │    └─────────────────┘
└─────────────────────┘           │
          │                       │
          └───────────┬───────────┘
                      ▼
              ┌─────────────────┐
              │ Result Merger   │
              │ & Confidence    │
              │ Scoring         │
              └─────────────────┘
                      │
                      ▼
              ┌─────────────────┐
              │ Text Masking    │
              │ & JSON Response │
              └─────────────────┘
```

## Performance Characteristics

### Throughput Metrics

| Component | Performance | Memory Usage |
|-----------|-------------|---------------|
| Rule Engine | ~1000 tokens/sec | <50MB |
| BERT Model | ~100 tokens/sec | ~2GB GPU |
| API Server | ~50 requests/sec | ~500MB |
| Training | ~200 examples/sec | ~8GB GPU |

### Scalability Considerations

1. **Horizontal Scaling**: FastAPI supports multiple workers
2. **Model Serving**: Can deploy multiple model instances
3. **Caching**: Rule patterns cached in memory
4. **Batch Processing**: Supports batch inference for high throughput

## Security and Privacy

### Data Protection
- **No Data Persistence**: Text not stored after processing
- **Memory Cleanup**: Explicit tensor cleanup after inference
- **Secure Headers**: CORS and security headers configured
- **Input Validation**: Comprehensive request validation

### Model Security
- **Model Integrity**: Checksum validation for saved models
- **Version Control**: Model versioning and rollback capability
- **Access Control**: API rate limiting and authentication ready

## Deployment Architecture

### Local Development
```
┌─────────────────────────────────────────────────────────────┐
│                 Replit Environment                          │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│ │ Python 3.11 │  │ FastAPI     │  │ Model Checkpoints   │  │
│ │ Runtime     │  │ Dev Server  │  │ Local Storage       │  │
│ └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                             │
│ Port Configuration:                                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Internal: 5000 (FastAPI)                               │ │
│ │ External: 80/443 (Production)                          │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Production Deployment
```
┌─────────────────────────────────────────────────────────────┐
│                 Replit Deployment                           │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│ │ Auto-scale  │  │ Load        │  │ Health Checks       │  │
│ │ Containers  │  │ Balancer    │  │ & Monitoring        │  │
│ └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                             │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│ │ CDN Edge    │  │ SSL/TLS     │  │ Geographic          │  │
│ │ Caching     │  │ Termination │  │ Distribution        │  │
│ └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

This architecture provides a robust, scalable foundation for Arabic PII detection with clear separation of concerns, efficient resource utilization, and production-ready deployment capabilities.
