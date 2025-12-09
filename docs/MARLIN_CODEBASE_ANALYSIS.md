# MARLIN Codebase Analysis

## Executive Summary

MARLIN (Methylation- and AI-guided Rapid Leukemia Subtype Inference) is a deep neural network-based system for classifying acute leukemia subtypes using DNA methylation profiles. The system is implemented in R with TensorFlow/Keras backend and is designed for both batch prediction and real-time analysis during Oxford Nanopore Technologies (ONT) sequencing runs.

---

## 1. What is MARLIN and What Does It Do?

### Purpose
MARLIN is a machine learning classifier that:
- **Classifies acute leukemia subtypes** based on DNA methylation patterns
- **Uses sparse methylation profiles** from nanopore sequencing data
- **Identifies 42 distinct methylation classes** representing different leukemia subtypes and lineages
- **Provides real-time predictions** during live sequencing runs
- **Aggregates methylation information** cumulatively to improve confidence over time

### Key Capabilities
1. **Real-time Classification**: Monitors sequencing output and provides running predictions
2. **Batch Processing**: Can predict from pre-computed methylation files
3. **Multiple Genome Assemblies**: Supports hg19, hg38, and T2T human genome references
4. **Progressive Accumulation**: Coverage improves as more sequencing data arrives
5. **Lineage Classification**: Groups predictions into lineages and methylation families
6. **Web Interface**: shinyMARLIN provides web-based access for non-technical users

### Clinical Application
The system assists in rapid diagnosis of acute leukemia by:
- Processing DNA methylation data from tumor samples
- Matching patterns against trained classification model
- Outputting probability scores for each leukemia class
- Highlighting the most probable classification with confidence metrics

---

## 2. Technology Stack

### Core Framework
- **R Programming Language** (v4.1.3)
- **TensorFlow** (v2.13) - Backend deep learning framework
- **Keras** (v2.13) - High-level neural network API for R
- **Python** (v3.10) - Required for TensorFlow/Keras backend

### Key R Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| keras | 2.13.0 | Deep learning model definition and training |
| tensorflow | 2.13 | Backend computation engine (GPU-enabled) |
| data.table | 1.14.2 | Fast data manipulation and I/O |
| doParallel | 1.0.17 | Parallel processing for multiple samples |
| foreach | 1.5.2 | Loop parallelization framework |
| openxlsx | Latest | Excel file reading (class annotations) |

### Supporting Tools
- **modkit**: Oxford Nanopore Technologies tool for extracting DNA modifications from BAM files
- **samtools**: BAM file manipulation and indexing
- **MinKNOW**: ONT basecalling and sequencing control software
- **Conda**: Environment management for reproducible setup

### Data Format Support
- **Input**: BED format methylation calls (chromosome, position, methylation beta values)
- **Output**: Prediction matrices (samples × 42 classes)
- **Model Storage**: HDF5 format (Hierarchical Data Format)
- **Configuration**: Excel (XLSX) for class annotations

---

## 3. Main Components for Training and Prediction

### Training Pipeline (`MARLIN_training.R`)

**Purpose**: Train the neural network on reference methylation array data

**Key Steps**:
1. **Data Loading**
   - Load reference beta values matrix (2,356 samples × 357,340 CpG sites)
   - Load class labels (42 methylation classes)
   - Merge peripheral blood controls into single category

2. **Data Preparation**
   - Binarize beta values (>0.5 → 1, ≤0.5 → -1)
   - Upsample to balanced dataset (50 samples per class)
   - Data augmentation: randomly flip 10% of CpG values per sample
   - Convert to one-hot encoded labels for multiclass

3. **Model Training**
   - 2,100 training samples (50 per class × 42 classes)
   - Input shape: 357,340 features (CpG sites)
   - 3,000 epochs with batch size 32
   - Learning rate: 10^-5 (Adam optimizer)
   - Loss function: Categorical cross-entropy
   - Metrics: Precision, Recall

4. **Model Persistence**
   - Save model: `nn.model.hdf5`
   - Save training history: `nn.model.history.RData`

**Example Command**:
```bash
CUDA_VISIBLE_DEVICES=1 Rscript MARLIN_training.R
```

### Prediction Pipeline (`MARLIN_prediction.R`)

**Purpose**: Batch prediction on new samples using trained model

**Key Steps**:
1. **Load Reference**
   - Load ordered CpG names (357,340 features)
   - Load trained Keras model from HDF5

2. **Process Input Files**
   - Read multiple BED format methylation files
   - **Parallel processing**: 24 cores by default (adjustable)
   - For each sample:
     - Extract methylation beta values
     - Match against reference CpG positions
     - Binarize values (>0.5 → 1, <0.5 → -1)
     - Handle missing data (NA → 0)

3. **Generate Predictions**
   - Convert to matrix format
   - Batch predict using neural network
   - Output: 42 class probability scores per sample

4. **Output Generation**
   - Row names: Sample file names
   - Column names: 42 methylation class names
   - Format: Probability scores (0-1)
   - Save as: `predictions.RData`

**Example Command**:
```bash
CUDA_VISIBLE_DEVICES=1 Rscript MARLIN_prediction.R
```

---

## 4. Model Architecture

### Neural Network Structure

```
Input Layer
    ├─ 357,340 nodes (one per CpG site)
    └─ Dropout (99% rate) - aggressive regularization
    
Hidden Layer 1
    ├─ 256 nodes
    ├─ Activation: Sigmoid
    └─ Fully connected
    
Hidden Layer 2
    ├─ 128 nodes
    ├─ Activation: Sigmoid
    └─ Fully connected
    
Output Layer
    ├─ 42 nodes (one per methylation class)
    └─ Activation: Softmax (multi-class probabilities)
```

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Loss Function | Categorical Cross-Entropy | Standard for multi-class classification |
| Optimizer | Adam (lr=10^-5) | Adaptive learning with very low rate for stability |
| Epochs | 3,000 | Sufficient for convergence on balanced dataset |
| Batch Size | 32 | Standard choice for moderate datasets |
| Dropout Rate | 99% | Extreme regularization on input layer |
| Input Binarization | Threshold 0.5 | Simplifies high-dimensional sparse input |
| Class Balancing | 50 samples/class | Uniform representation across all classes |

### Model Characteristics
- **Total Parameters**: ~186M (heavily regularized)
- **Sparsity Handling**: NA/missing values coded as 0
- **Input Dimensionality**: 357,340 (ultra-high dimensional)
- **Output Dimensionality**: 42 (multi-class classification)
- **Model Size**: ~1.1 GB (HDF5 format)

---

## 5. Data Formats

### Input Data Format: BED (Browser Extended) Format

**Standard BED Format for Methylation Calls**:
```
chromosome  start_position  end_position  methylation_beta_value  probe_id
chr1        10000           10001         0.75                    cg21870274
chr1        10500           10501         NA                      cg12345678
chrX        15000           15001         0.25                    cg98765432
```

**Specifications**:
- Tab-delimited text file
- Column 1: Chromosome (e.g., chr1, chrX)
- Column 2: Start position (0-based)
- Column 3: End position
- Column 4: Methylation beta value (0-1 or NA for uncovered)
- Column 5: Probe ID (e.g., cg-notation from Illumina)

### Reference Data Formats

#### 1. Feature Set (`marlin_v1.features.RData`)
- **Format**: Compressed R object (1.7 MB)
- **Contents**: `betas_sub_names` vector
- **Description**: Ordered list of 357,340 reference CpG probe IDs
- **Purpose**: Ensures consistent feature ordering for predictions

#### 2. Probe Coordinates (`marlin_v1.probes_*.bed.gz`)
- **Format**: Gzip-compressed BED format
- **Size**: ~5-6 MB per genome version
- **Versions**: hg19 (27.7 MB), hg38 (27.7 MB), T2T (27.7 MB)
- **Contents**: Genomic coordinates for ~370,000 reference probes

#### 3. Class Annotations (`marlin_v1.class_annotations.xlsx`)
- **Format**: Excel 2007+ spreadsheet (22 KB)
- **Columns**:
  - `model_id`: Numeric class identifier (0-41)
  - `class_name_current`: Full methylation class name
  - `lineage`: Leukemia lineage grouping
  - `mcf`: Methylation class family
  - `color_lineage`: Display color for lineage
  - `color_mcf`: Display color for family
  - `color_mc`: Display color for class
- **Purpose**: Maps model output to human-readable classifications

#### 4. Trained Model (`marlin_v1.model.hdf5`)
- **Format**: HDF5 (Hierarchical Data Format, version 5)
- **Size**: 1.1 GB
- **Contents**: Serialized Keras neural network with all weights
- **Compatibility**: Requires Keras 2.13+ to load

#### 5. Training Data (`betas.RData`, `y.RData`)
- **Format**: R binary serialized format
- **betas.RData**: Matrix (2,356 samples × 357,340 CpGs)
- **y.RData**: Vector of class labels (42 categories)
- **Purpose**: Source data for model training

### Output Data Formats

#### 1. Prediction Matrix (`.RData`)
- **Format**: R object containing matrix
- **Dimensions**: N samples × 42 classes
- **Values**: Softmax probabilities (0-1, sum to 1)
- **Naming**: Row names = sample file names, Col names = class names

#### 2. Text Predictions (`.pred.txt`)
- **Format**: Tab-delimited text
- **Columns**: 42 class scores + `cov_cpgs` + `time`
- **Headers**: Class names, "cov_cpgs", "time"
- **Purpose**: Human-readable output and accumulation tracking

#### 3. Visualization (`.pred.pdf`)
- **Format**: PDF document
- **Contents**: 8 plots per prediction:
  - Coverage over time
  - Lineage scores (time series + final barplot)
  - Methylation family scores (time series + final barplot)
  - Methylation class scores (time series + final barplot)
- **Color coding**: Based on class annotations

### Data Transformation Pipeline

```
Raw ONT Sequencing (FASTQ)
    ↓
Basecalling + Modification Detection (MinKNOW)
    ↓
BAM files with Mod Tags
    ↓
modkit pileup → Methylation Pileup
    ↓
Process & Merge (2_process_pileup.R)
    ↓
Cumulative BED File (calls.bam.pileup.bed)
    ↓
Match to Reference (betas_sub_names)
    ↓
Binarize + Handle Missing Data
    ↓
MARLIN Model Prediction
    ↓
Probability Scores (42 classes)
    ↓
Visualization & Output
```

---

## 6. Key Files and Their Purposes

### Core Implementation Files

| File | Language | Purpose | Lines |
|------|----------|---------|-------|
| `MARLIN_training.R` | R | Train the neural network model | 133 |
| `MARLIN_prediction.R` | R | Batch prediction on new samples | 78 |
| `MARLIN_realtime/0_real_time_prediction_main.sh` | Bash | Main loop for real-time monitoring | 14 |
| `MARLIN_realtime/1_process_live.sh` | Bash | Process individual BAM files | 20 |
| `MARLIN_realtime/2_process_pileup.R` | R | Merge pileup files | 45 |
| `MARLIN_realtime/3_marlin_predictions_live.R` | R | Generate predictions from pileup | 45 |
| `MARLIN_realtime/4_plot_live2.R` | R | Create visualization plots | 89 |

### Data Files

| File | Format | Size | Purpose |
|------|--------|------|---------|
| `betas.RData` | R Binary | ~200 MB | Training beta values (2,356 samples) |
| `y.RData` | R Binary | ~10 KB | Training class labels |
| `marlin_v1.features.RData` | R Compressed | 1.7 MB | Reference CpG names (357,340 features) |
| `marlin_v1.model.hdf5` | HDF5 | 1.1 GB | Trained Keras neural network |
| `marlin_v1.probes_hg19.bed.gz` | Gzip BED | 6.1 MB | hg19 probe coordinates |
| `marlin_v1.probes_hg38.bed.gz` | Gzip BED | 5.4 MB | hg38 probe coordinates |
| `marlin_v1.probes_t2t.bed.gz` | Gzip BED | 5.4 MB | T2T probe coordinates |
| `marlin_v1.class_annotations.xlsx` | Excel | 22 KB | Class metadata and color schemes |

### Documentation Files

| File | Format | Purpose |
|------|--------|---------|
| `README.md` | Markdown | Installation and usage instructions |
| `MARLIN_realtime/README.txt` | Plain Text | Real-time prediction setup guide |
| `shinyMARLIN/README.md` | Markdown | Web interface documentation |
| `LICENSE` | Text | MIT license |

---

## 7. Workflow and Data Flow

### Real-Time Prediction Workflow

```
1. MONITORING (0_real_time_prediction_main.sh)
   - Polls directory every 5 seconds
   - Detects new .bam files
   - No .pred.pdf file yet = not processed
   
2. INDEXING & PILEUP (1_process_live.sh)
   - samtools index: Index BAM file
   - modkit pileup: Extract methylation from BAM
   - Creates .pileup file (methylation per position)
   
3. MERGE PILEUPS (2_process_pileup.R)
   - Combines all .pileup files to date
   - Aggregates methylation calls
   - Outputs: calls.bam.pileup.bed (cumulative)
   
4. PREDICT (3_marlin_predictions_live.R)
   - Load reference feature names
   - Match pileup to reference probes
   - Binarize values
   - Run MARLIN model inference
   - Output: .pred.RData and .pred.txt
   
5. VISUALIZE (4_plot_live2.R)
   - Load all previous predictions
   - Sort by timestamp
   - Create 8-panel PDF:
     * Coverage metrics
     * Lineage scores over time
     * Family scores over time
     * Class scores over time
   - Output: .pred.pdf
```

### Batch Prediction Workflow

```
1. INPUT
   - Multiple BED files with methylation calls
   - Each file = one sample
   
2. LOAD REFERENCE
   - Keras model from HDF5
   - Feature names from .RData
   
3. PARALLEL PROCESSING
   - For each input file (parallel, 24 cores):
     * Read BED file
     * Match to reference features
     * Binarize values
     * Handle missing data
     
4. PREDICT
   - Stack all samples into matrix
   - Single batch inference through Keras
   
5. OUTPUT
   - Predictions matrix (N samples × 42 classes)
   - Save to .RData
```

---

## 8. System Components Breakdown

### Component 1: Data Acquisition
- **Source**: Oxford Nanopore sequencing runs
- **Tool**: modkit for methylation extraction
- **Output**: BAM files with modification tags

### Component 2: Data Processing
- **Scripts**: `1_process_live.sh`, `2_process_pileup.R`
- **Operations**: Indexing, pileup generation, merging
- **Output**: Cumulative methylation BED files

### Component 3: Feature Extraction
- **Reference**: 357,340 CpG sites
- **Matching**: Align sample methylation to reference
- **Encoding**: Binarization (1/-1) and NA handling

### Component 4: Inference
- **Model**: Keras neural network (1.1 GB)
- **Input**: 357,340 features (binarized)
- **Output**: 42 class probabilities

### Component 5: Visualization & Output
- **PDF Reports**: Time-series plots with final predictions
- **Text Output**: Tab-delimited prediction scores
- **Color Coding**: By lineage, family, and class

### Component 6: Web Interface
- **shinyMARLIN**: R Shiny-based web application
- **Access**: https://hovestadtlab.shinyapps.io/shinyMARLIN/
- **Input**: BedMethyl format files
- **Users**: Non-technical clinicians

---

## 9. Key Technical Features

### Sparse Input Handling
- Missing methylation data coded as 0 (not covered)
- Allows handling of incomplete methylation profiles
- Progressive accumulation improves coverage

### Regularization Strategy
- 99% dropout on input layer (extreme regularization)
- Prevents overfitting on high-dimensional sparse input
- Sigmoid activations on hidden layers

### Parallel Processing
- Default: 24 CPU cores for sample processing
- Implemented via `doParallel` and `foreach`
- Significant speedup for batch predictions

### Multi-Reference Support
- hg19, hg38, T2T probes available
- Users select appropriate genome version
- Coordinates provided in separate BED files

### GPU Acceleration
- TensorFlow/Keras can use CUDA GPUs
- Controlled via `CUDA_VISIBLE_DEVICES` environment variable
- Significant speedup for training and inference

---

## 10. Requirements and Dependencies

### System Requirements
- R 4.1.3 (specific version recommended)
- Python 3.10
- CUDA toolkit (optional, for GPU acceleration)
- 2+ GB RAM (16+ GB for training)

### Installation Command
```bash
conda create --name marlin -c conda-forge r-base=4.1.3
conda activate marlin
conda install -c conda-forge r-keras=2.13 r-tensorflow=2.13 tensorflow-gpu=2.13 python=3.10
```

### Required External Tools
- modkit (ONT methylation extraction)
- samtools (BAM manipulation)
- MinKNOW (ONT sequencing control)

---

## Summary

MARLIN is a sophisticated machine learning system designed specifically for rapid acute leukemia classification using DNA methylation profiles from Oxford Nanopore sequencing. Its architecture elegantly combines high-dimensional sparse genomic data processing with a carefully regularized deep neural network to achieve reliable classification into 42 distinct leukemia subtypes. The system offers both real-time streaming predictions during live sequencing and batch processing capabilities, along with an accessible web interface for clinical use.

The codebase is well-organized, professionally documented, and optimized for both computational efficiency (parallel processing, GPU support) and practical usability in clinical settings.
